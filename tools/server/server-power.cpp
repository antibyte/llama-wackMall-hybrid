#include "server-power.h"
#include "server-common.h"

#include "ggml.h"

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

#ifndef _WIN32
#include <dirent.h>
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

extern char ** environ;
#endif

struct server_power::impl {
    std::string busy_cmd;
    std::string idle_cmd;
    int         idle_delay_ms = 0;
    bool        enabled       = false;

    std::string dir;
    std::string marker;

    std::mutex              mtx;
    std::condition_variable cv;
    std::thread             worker;

    bool busy       = false; // this process holds its busy marker
    bool idle_armed = false;
    bool stop       = false;
    std::chrono::steady_clock::time_point idle_deadline;

    // busy && !idle_armed, read without the mutex on every update
    std::atomic<bool> fast_busy { false };

#ifndef _WIN32
    void run(const std::string & cmd, const char * what) {
        if (cmd.empty()) {
            return;
        }
        const int64_t t0 = ggml_time_us();
        const char * argv[] = { "/bin/sh", "-c", cmd.c_str(), nullptr };
        pid_t pid = 0;
        if (posix_spawn(&pid, "/bin/sh", nullptr, nullptr, const_cast<char **>(argv), environ) != 0) {
            SRV_WRN("power: failed to start %s command\n", what);
            return;
        }
        int status = 0;
        while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
        const double ms = (ggml_time_us() - t0) / 1000.0;
        if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
            SRV_INF("power: %s command done in %.1f ms\n", what, ms);
        } else {
            SRV_WRN("power: %s command failed (status %d) after %.1f ms\n", what, status, ms);
        }
    }

    // counts the live busy markers of other processes and removes stale ones
    int count_other_busy() {
        DIR * d = opendir(dir.c_str());
        if (d == nullptr) {
            return 0;
        }
        int n = 0;
        const pid_t self = getpid();
        while (struct dirent * e = readdir(d)) {
            int pid = 0;
            if (sscanf(e->d_name, "busy.%d", &pid) != 1 || pid <= 0 || pid == self) {
                continue;
            }
            if (kill(pid, 0) == 0 || errno == EPERM) {
                n++;
            } else {
                unlink((dir + "/" + e->d_name).c_str());
            }
        }
        closedir(d);
        return n;
    }

    // to_busy: create our marker, run busy_cmd if no other server is busy
    // otherwise: remove our marker, run idle_cmd if no other server is busy
    // to_busy < 0: only apply idle_cmd when nobody is busy (startup)
    void transition(int to_busy) {
        const int fd = open((dir + "/lock").c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
        if (fd < 0) {
            SRV_WRN("power: cannot open %s/lock\n", dir.c_str());
            return;
        }
        while (flock(fd, LOCK_EX) < 0 && errno == EINTR) {}

        const int others = count_other_busy();
        if (to_busy > 0) {
            const int mfd = open(marker.c_str(), O_WRONLY | O_CREAT | O_CLOEXEC, 0600);
            if (mfd >= 0) {
                close(mfd);
            }
            if (others == 0) {
                run(busy_cmd, "busy");
            }
        } else {
            if (to_busy == 0) {
                unlink(marker.c_str());
            }
            if (others == 0) {
                run(idle_cmd, "idle");
            }
        }

        flock(fd, LOCK_UN);
        close(fd);
    }

    bool init_dir() {
        const char * env = getenv("LLAMA_POWER_DIR");
        dir = env && *env ? env : "/tmp/llama-power-" + std::to_string(getuid());
        if (mkdir(dir.c_str(), 0700) != 0 && errno != EEXIST) {
            return false;
        }
        struct stat st;
        if (lstat(dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode) || st.st_uid != getuid()) {
            return false;
        }
        marker = dir + "/busy." + std::to_string(getpid());
        return true;
    }
#else
    void transition(int) {}
    bool init_dir() { return false; }
#endif

    void worker_main() {
        std::unique_lock<std::mutex> lock(mtx);
        while (!stop) {
            if (!idle_armed) {
                cv.wait(lock, [this] { return stop || idle_armed; });
                continue;
            }
            const auto deadline = idle_deadline;
            if (cv.wait_until(lock, deadline, [&] { return stop || !idle_armed || idle_deadline != deadline; })) {
                continue;
            }
            idle_armed = false;
            busy = false;
            transition(0);
        }
    }
};

server_power::server_power(const std::string & busy_cmd, const std::string & idle_cmd, int idle_delay_ms)
    : pimpl(std::make_unique<impl>()) {
    if (busy_cmd.empty() && idle_cmd.empty()) {
        return;
    }
    pimpl->busy_cmd      = busy_cmd;
    pimpl->idle_cmd      = idle_cmd;
    pimpl->idle_delay_ms = idle_delay_ms;
    if (!pimpl->init_dir()) {
        SRV_WRN("%s", "power: profile switching is unavailable (no usable state directory)\n");
        return;
    }
    pimpl->enabled = true;
    SRV_INF("power: profile switching enabled, idle delay %d ms, state in %s\n", idle_delay_ms, pimpl->dir.c_str());

    // nothing works yet: apply the idle profile unless another server is busy
    pimpl->transition(-1);
    pimpl->worker = std::thread([this] { pimpl->worker_main(); });
}

server_power::~server_power() {
    if (!pimpl->enabled) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(pimpl->mtx);
        pimpl->stop = true;
    }
    pimpl->cv.notify_one();
    pimpl->worker.join();
    if (pimpl->busy) {
        pimpl->transition(0);
    }
}

void server_power::busy() {
    if (!pimpl->enabled || pimpl->fast_busy.load(std::memory_order_acquire)) {
        return;
    }
    std::lock_guard<std::mutex> lock(pimpl->mtx);
    pimpl->idle_armed = false;
    if (!pimpl->busy) {
        pimpl->transition(1);
        pimpl->busy = true;
    }
    pimpl->fast_busy.store(true, std::memory_order_release);
    pimpl->cv.notify_one();
}

void server_power::idle() {
    if (!pimpl->enabled || !pimpl->fast_busy.load(std::memory_order_acquire)) {
        return;
    }
    std::lock_guard<std::mutex> lock(pimpl->mtx);
    pimpl->fast_busy.store(false, std::memory_order_release);
    pimpl->idle_armed    = true;
    pimpl->idle_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(pimpl->idle_delay_ms);
    pimpl->cv.notify_one();
}

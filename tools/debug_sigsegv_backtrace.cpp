#include <execinfo.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ucontext.h>
#include <unistd.h>

namespace {

alignas(16) unsigned char alt_stack[64 * 1024];

void write_literal(const char * text) {
    const size_t size = strlen(text);
    ssize_t written = 0;
    while ((size_t) written < size) {
        const ssize_t result = write(STDERR_FILENO, text + written, size - written);
        if (result <= 0) {
            break;
        }
        written += result;
    }
}

void handle_fault(int signal_number, siginfo_t * info, void * context_ptr) {
    char header[512];
    int header_size = 0;

#if defined(__x86_64__)
    const ucontext_t * context = static_cast<const ucontext_t *>(context_ptr);
    const greg_t * registers = context->uc_mcontext.gregs;
    header_size = snprintf(
            header,
            sizeof(header),
            "\nHYBRID_DEBUG_SIGNAL signal=%d fault=%p rip=0x%llx rsp=0x%llx "
            "rdi=0x%llx rsi=0x%llx rdx=0x%llx rcx=0x%llx\n",
            signal_number,
            info ? info->si_addr : nullptr,
            (unsigned long long) registers[REG_RIP],
            (unsigned long long) registers[REG_RSP],
            (unsigned long long) registers[REG_RDI],
            (unsigned long long) registers[REG_RSI],
            (unsigned long long) registers[REG_RDX],
            (unsigned long long) registers[REG_RCX]);
#else
    (void) context_ptr;
    header_size = snprintf(
            header,
            sizeof(header),
            "\nHYBRID_DEBUG_SIGNAL signal=%d fault=%p\n",
            signal_number,
            info ? info->si_addr : nullptr);
#endif

    if (header_size > 0) {
        const size_t size = (size_t) header_size < sizeof(header) ? (size_t) header_size : sizeof(header) - 1;
        ssize_t written = 0;
        while ((size_t) written < size) {
            const ssize_t result = write(STDERR_FILENO, header + written, size - written);
            if (result <= 0) {
                break;
            }
            written += result;
        }
    }

    void * frames[128];
    const int frame_count = backtrace(frames, (int) (sizeof(frames) / sizeof(frames[0])));
    write_literal("HYBRID_DEBUG_BACKTRACE_BEGIN\n");
    backtrace_symbols_fd(frames, frame_count, STDERR_FILENO);
    write_literal("HYBRID_DEBUG_BACKTRACE_END\n");
    _exit(128 + signal_number);
}

__attribute__((constructor)) void install_fault_handler() {
    stack_t stack = {};
    stack.ss_sp = alt_stack;
    stack.ss_size = sizeof(alt_stack);
    const bool have_alt_stack = sigaltstack(&stack, nullptr) == 0;

    struct sigaction action = {};
    action.sa_sigaction = handle_fault;
    sigemptyset(&action.sa_mask);
    action.sa_flags = SA_SIGINFO | SA_RESETHAND | (have_alt_stack ? SA_ONSTACK : 0);
    sigaction(SIGSEGV, &action, nullptr);
    sigaction(SIGBUS, &action, nullptr);
}

} // namespace

#pragma once

#include <memory>
#include <string>

// Runs busy_cmd before model work starts and idle_cmd once the server has been
// idle for idle_delay_ms. The busy state is shared by all servers of the same
// user through marker files, so idle_cmd only runs when no server is working.
class server_power {
public:
    server_power(const std::string & busy_cmd, const std::string & idle_cmd, int idle_delay_ms);
    ~server_power();

    server_power(const server_power &) = delete;
    server_power & operator=(const server_power &) = delete;

    // blocks until busy_cmd has finished when this is the first busy server
    void busy();

    // arms the delayed idle transition; a later busy() cancels it
    void idle();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

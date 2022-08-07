#pragma once
#include <chrono>
#include <iostream>
using namespace std;
using namespace chrono;

namespace lxh {
class Timer {
public:
    system_clock::time_point start_time;
    system_clock::time_point end_time;
    bool printIt;
    Timer(bool _printIt=true)
    {
        printIt = _printIt;
    }
    void start()
    {
        start_time = system_clock::now();
    }
    void end(const string msg, bool highlightMsg = false, bool higlightSec = true)
    {

        if (printIt) {
            end_time = system_clock::now();
            auto duration = duration_cast<microseconds>(end_time - start_time);
            start_time = end_time; // 防止重复调用end
            if (highlightMsg)
                cout << "\033[91m" << msg << "\033[30m";
            else
                cout << msg;
            cout << " cost ";
            if (higlightSec)
                cout << "\033[36m" << double(duration.count()) * microseconds::period::num / microseconds::period::den
                     << "\033[30m";
            else
                cout << double(duration.count()) * microseconds::period::num / microseconds::period::den;
            cout << " second" << endl;
        }
    }
};

}
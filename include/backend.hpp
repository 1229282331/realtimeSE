#pragma once
#include <iostream>
#include <string>
#include <vector>
#include "signal.h"

class Backend{
public:
    Backend() = default;
    ~Backend() = default;

    virtual SE::Vectorf forward(const SE::Vectorf& input) = 0;

private:
    virtual void init_caches() = 0;
};

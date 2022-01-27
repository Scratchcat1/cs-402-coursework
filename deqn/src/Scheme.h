#ifndef SCHEME_H_
#define SCHEME_H_

#include "Mesh.h"

class Scheme {
    private:
        Mesh* mesh;
    public:
        virtual void doAdvance(const float dt) = 0;

        virtual void init() = 0;
};
#endif

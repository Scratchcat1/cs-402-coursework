#ifndef EXPLICIT_SCHEME_SINGLE_THREAD_H_
#define EXPLICIT_SCHEME_SINGLE_THREAD_H_

#include "Scheme.h"
#include "InputFile.h"

class ExplicitSchemeSingleThread : public Scheme {
    private:
        Mesh* mesh;

        void updateBoundaries();
        void reset();
        void diffuse(double dt);
        void reflectBoundaries(int boundary_id);
    public:
        ExplicitSchemeSingleThread(const InputFile* input, Mesh* m);

        void doAdvance(const double dt);

        void init();
};
#endif

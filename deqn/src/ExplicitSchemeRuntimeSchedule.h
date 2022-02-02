#ifndef EXPLICIT_SCHEME_RUNTIME_SCHEDULE_H_
#define EXPLICIT_SCHEME_RUNTIME_SCHEDULE_H_

#include "Scheme.h"
#include "InputFile.h"

class ExplicitSchemeRuntimeSchedule : public Scheme {
    private:
        void updateBoundaries();
        void reset();
        void diffuse(double dt);
        void reflectBoundaries();
    protected:
        Mesh* mesh;
    public:
        ExplicitSchemeRuntimeSchedule(const InputFile* input, Mesh* m);

        void doAdvance(const double dt);

        void init();
};
#endif

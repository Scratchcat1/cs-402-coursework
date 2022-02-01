#ifndef EXPLICIT_SCHEME_TILES_H_
#define EXPLICIT_SCHEME_TILES_H_

#include "ExplicitScheme.h"
#include "InputFile.h"

class ExplicitSchemeTiles : public ExplicitScheme {
    private:
        int tile_size;

        void diffuse(double dt);
    public:
        ExplicitSchemeTiles(const InputFile* input, Mesh* m, int t_size);
};
#endif

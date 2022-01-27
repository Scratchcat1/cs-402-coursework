#ifndef VTKWRITER_H_
#define VTKWRITER_H_

#include <string>
#include "Mesh.h"

class VtkWriter {
    private:
        std::string dump_basename;

        std::string vtk_header;

        Mesh* mesh;

        void writeVtk(int step, float time);
    public:
        VtkWriter(std::string basename, Mesh* mesh);

        void write(int step, float time);
};
#endif

#ifndef INPUT_FILE_H_
#define INPUT_FILE_H_

#include <map>
#include <string>
#include <vector>

class InputFile {
    private:
        std::map<std::string, std::string> pairs;

        template <typename T> T get(
                const std::string& name,
                const T& dfault) const;
    public:
        InputFile(const char* filename);
        ~InputFile();

        int getInt(
                const std::string& name,
                const int dfault) const;

        float getfloat(
                const std::string& name,
                const float dfault) const;

        std::string getString(
                const std::string& name,
                const std::string& dfault) const;

        std::vector<float> getfloatList(
                const std::string& name,
                const std::vector<float>& dfault) const;
};
#endif

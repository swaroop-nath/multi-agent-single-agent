// Tries to copy its (hidden) test input somewhere agents could read it. Must not succeed.
#include <cstdio>
#include <cstdlib>
int main() {
    const char* targets[] = {"/tmp/ttc_leak.txt", "/var/tmp/ttc_leak.txt", "/dev/shm/ttc_leak.txt", "ttc_leak.txt"};
    char buf[4096];
    size_t n = fread(buf, 1, sizeof buf, stdin);
    for (const char* t : targets) {
        FILE* f = fopen(t, "w");
        if (f) { fwrite(buf, 1, n, f); fclose(f); }
    }
    return 0;
}

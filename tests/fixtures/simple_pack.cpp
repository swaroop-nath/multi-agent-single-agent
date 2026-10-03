// Trivially valid polyomino packing: every piece in its own column block, side by side.
#include <cstdio>
#include <vector>
#include <algorithm>
int main() {
    int n;
    if (scanf("%d", &n) != 1) return 0;
    std::vector<long long> X(n), Y(n);
    long long cursor = 0, H = 1;
    for (int i = 0; i < n; i++) {
        int k;
        scanf("%d", &k);
        long long minx = 1e18, miny = 1e18, maxx = -1e18, maxy = -1e18;
        for (int j = 0; j < k; j++) {
            long long x, y;
            scanf("%lld %lld", &x, &y);
            minx = std::min(minx, x); maxx = std::max(maxx, x);
            miny = std::min(miny, y); maxy = std::max(maxy, y);
        }
        X[i] = cursor - minx;
        Y[i] = -miny;
        cursor += maxx - minx + 1;
        H = std::max(H, maxy - miny + 1);
    }
    printf("%lld %lld\n", cursor, H);
    for (int i = 0; i < n; i++) printf("%lld %lld 0 0\n", X[i], Y[i]);
}

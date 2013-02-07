#include <vector>
#include <iostream>
#include <fstream>
#include <cstdlib>

/*
 * Generates problem file for poisson equation in a unit square.
 */
using namespace std;

int main(int argc, char *argv[]) {
    int    n   = argc > 1 ? atoi(argv[1]) : 64;
    int    n3  = n * n * n;
    double h2i = (n - 1) * (n - 1);

    vector<int>    row;
    vector<int>    col;
    vector<double> val;
    vector<double> rhs;

    row.reserve(n3 + 1);
    col.reserve(7 * n3);
    val.reserve(7 * n3);
    rhs.reserve(n3);

    row.push_back(0);

    for(int k = 0, idx = 0; k < n; ++k) {
        for(int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i, ++idx) {
                if (
                        i == 0 || i == n - 1 ||
                        j == 0 || j == n - 1 ||
                        k == 0 || k == n - 1
                   )
                {
                    col.push_back(idx);
                    val.push_back(1);

                    rhs.push_back(0);
                } else {
                    col.push_back(idx - n * n);
                    val.push_back(-h2i);

                    col.push_back(idx - n);
                    val.push_back(-h2i);

                    col.push_back(idx - 1);
                    val.push_back(-h2i);

                    col.push_back(idx);
                    val.push_back(6 * h2i);

                    col.push_back(idx + 1);
                    val.push_back(-h2i);

                    col.push_back(idx + n);
                    val.push_back(-h2i);

                    col.push_back(idx + n * n);
                    val.push_back(-h2i);

                    rhs.push_back(1);
                }

                row.push_back(col.size());
            }
        }
    }

    ofstream f("problem.dat", ios::binary);

    f.write((char*)&n3, sizeof(n));
    f.write((char*)row.data(), row.size() * sizeof(row[0]));
    f.write((char*)col.data(), col.size() * sizeof(col[0]));
    f.write((char*)val.data(), val.size() * sizeof(val[0]));
    f.write((char*)rhs.data(), rhs.size() * sizeof(rhs[0]));

    cout << "Wrote \"problem.dat\"" << endl;
}

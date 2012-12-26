#include <vector>
#include <iostream>
#include <fstream>
#include <cstdlib>

/*
 * Generates problem file for poisson equation in a unit square.
 */
using namespace std;

int main(int argc, char *argv[]) {
    int    n   = argc > 1 ? atoi(argv[1]) : 1024;
    int    n2  = n * n;
    double h   = 1.0 / (n - 1);
    double h2i = (n - 1) * (n - 1);

    vector<int>    row;
    vector<int>    col;
    vector<double> val;
    vector<double> rhs;

    row.reserve(n2 + 1);
    col.reserve(5 * n2);
    val.reserve(5 * n2);
    rhs.reserve(n2);

    row.push_back(0);

    for (int i = 0, idx = 0; i < n; ++i) {
        double x = i * h;
        for(int j = 0; j < n; ++j, ++idx) {
            double y = j * h;
            if (
                    i == 0 || i == n - 1 ||
                    j == 0 || j == n - 1
               )
            {
                col.push_back(idx);
                val.push_back(1);

                rhs.push_back(0);
            } else {
                col.push_back(idx - n);
                val.push_back(-h2i);

                col.push_back(idx - 1);
                val.push_back(-h2i);

                col.push_back(idx);
                val.push_back(4 * h2i);

                col.push_back(idx + 1);
                val.push_back(-h2i);

                col.push_back(idx + n);
                val.push_back(-h2i);

                rhs.push_back( 2 * (x - x * x + y - y * y) );
            }

            row.push_back(col.size());
        }
    }

    ofstream f("problem.dat", ios::binary);

    f.write((char*)&n2, sizeof(n));
    f.write((char*)row.data(), row.size() * sizeof(row[0]));
    f.write((char*)col.data(), col.size() * sizeof(col[0]));
    f.write((char*)val.data(), val.size() * sizeof(val[0]));
    f.write((char*)rhs.data(), rhs.size() * sizeof(rhs[0]));

    cout << "Wrote \"problem.dat\"" << endl;
}

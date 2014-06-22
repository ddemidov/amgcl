#ifndef AMGCL_ADAPTER_EPETRA_HPP
#define AMGCL_ADAPTER_EPETRA_HPP

/*
The MIT License

Copyright (c) 2012-2014 Denis Demidov <dennis.demidov@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
\file    amgcl/adapter/epetra.hpp
\author  Denis Demidov <dennis.demidov@gmail.com>
\brief   Adapt Epetra_CrsMatrix from Trilinos.
\ingroup adapters
*/

#include <vector>

#include <Epetra_CrsMatrix.h>

#include <amgcl/backend/interface.hpp>

namespace amgcl {
namespace backend {

//---------------------------------------------------------------------------
// Specialization of matrix interface
//---------------------------------------------------------------------------
template <>
struct value_type < Epetra_CrsMatrix > {
    typedef double type;
};

template <>
struct rows_impl < Epetra_CrsMatrix > {
    static size_t get(
            const Epetra_CrsMatrix &A
            )
    {
        return A.NumMyRows();
    }
};

template <>
struct cols_impl < Epetra_CrsMatrix > {
    static size_t get(
            const Epetra_CrsMatrix &A
            )
    {
        return A.NumMyCols();
    }
};

template <>
struct nonzeros_impl < Epetra_CrsMatrix > {
    static size_t get(
            const Epetra_CrsMatrix &A
            )
    {
        return A.NumMyNonzeros();
    }
};

template <>
struct row_iterator < Epetra_CrsMatrix > {
    class type {
        public:
            typedef int    col_type;
            typedef double val_type;

            type(const Epetra_CrsMatrix &A, int row)
                : A(A)
            {
                int nnz;
                A.ExtractMyRowView(A.LRID(row + A.RowMap().MinMyGID()),
                        nnz, m_val, m_col);
                m_end = m_col + nnz;
            }

            operator bool() const {
                return m_col != m_end;
            }

            type& operator++() {
                ++m_col;
                ++m_val;
                return *this;
            }

            col_type col() const {
                return A.GCID(*m_col);
            }

            val_type value() const {
                return *m_val;
            }

        private:
            const Epetra_CrsMatrix &A;
            col_type * m_col;
            col_type * m_end;
            val_type * m_val;
    };
};

template <>
struct row_begin_impl< Epetra_CrsMatrix >
{
    static typename row_iterator< Epetra_CrsMatrix >::type
    get(const Epetra_CrsMatrix &A, size_t row) {
        return typename row_iterator<Epetra_CrsMatrix>::type(A, row);
    }
};

} // namespace backend
} // namespace amgcl


#endif

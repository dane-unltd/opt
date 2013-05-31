package opt

import (
	"github.com/dane-unltd/linalg/clapack"
	"github.com/dane-unltd/linalg/matrix"
	"github.com/kortschak/cblas"
	"testing"
)

type matops struct {
	cblas.Blas
	clapack.Lapack
}

func init() {
	matrix.Register(matops{})
}

func TestLinprog(t *testing.T) {
	m := 5
	n := 10
	tol := 1e-8

	A := matrix.RandN(m, n)
	At := A.TrView()
	b := matrix.NewVec(m)
	c := matrix.RandVec(n)
	xt := matrix.RandVec(n)
	b.Mul(A, xt)

	x, y, s := linprog(c, A, b, tol)

	rd := matrix.NewVec(n)
	rp := matrix.NewVec(m)
	rs := matrix.NewVec(n)

	rd.Sub(c, s)
	rd.AddMul(At, y, 1)
	rp.Mul(A, x)
	rp.Sub(b, rp)
	rs.MulH(x, s)
	rs.Neg(rs)
	dev := (rd.Asum() + rp.Asum() + rs.Asum()) / float64(n)
	if dev > tol {
		t.Log(dev)
		t.Fail()
	}
}

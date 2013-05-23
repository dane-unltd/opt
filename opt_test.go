package opt

import (
	"github.com/dane-unltd/linalg/clapack"
	. "github.com/dane-unltd/linalg/matrix"
	"github.com/kortschak/cblas"
	"math"
	"testing"
)

type cblasops struct {
	cblas.Blas
	clapack.Lapack
}

func TestSteepestDescent(t *testing.T) {
	n := 30
	Register(cblasops{})
	xStar := NewVec(n)
	xStar.AddSc(1)
	A := RandN(n)
	At := A.TrView()
	AtA := NewDense(n)
	AtA.Mul(At, A)

	bTmp := NewVec(n)
	bTmp.Mul(A, xStar)
	b := NewVec(n)
	b.Mul(At, bTmp)
	b.Scal(-2)

	c := bTmp.Nrm2Sq()
	obj := makeQuadratic(AtA, b, c)
	op := Problem{Obj: obj}

	x := NewVec(n)

	f := SteepestDescent(op, x, 1e-4, 500)

	if math.Abs(f) > 0.01 {
		t.Log(f)
		t.Log(obj.F(xStar))
		t.Fail()
	}
}

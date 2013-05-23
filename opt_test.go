package opt

import (
	"fmt"
	"github.com/dane-unltd/linalg/matrix"
	"math"
	"testing"
)

func TestSteepestDescent(t *testing.T) {
	n := 50
	xStar := matrix.NewVec(n)
	xStar.AddSc(1)
	A := matrix.RandN(n)
	At := A.TrView()
	AtA := matrix.NewDense(n)
	AtA.Mul(At, A)

	bTmp := matrix.NewVec(n)
	bTmp.Mul(A, xStar)
	b := matrix.NewVec(n)
	b.Mul(At, bTmp)
	b.Scal(-2)

	c := bTmp.Nrm2Sq()
	obj := makeQuadratic(AtA, b, c)

	x := matrix.NewVec(n)
	f, i1 := SteepestDescent(obj, x, 1e-4, 50000, Inexact)
	fmt.Println(i1)

	x = matrix.NewVec(n)
	f, i2 := SteepestDescent(obj, x, 1e-4, 50000, Exact)
	fmt.Println(i2)

	if math.Abs(f) > 0.01 {
		t.Log(f)
		t.Log(obj.F(xStar))
		t.Fail()
	}
}

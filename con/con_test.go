package con

import (
	"fmt"
	"github.com/dane-unltd/linalg/clapack"
	"github.com/dane-unltd/linalg/mat"
	"github.com/dane-unltd/opt"
	"github.com/dane-unltd/opt/linesearch"
	"github.com/kortschak/cblas"
	"math"
	"math/rand"
	"testing"
)

var cops struct {
	cblas.Blas
	clapack.Lapack
}

func TestQuadratic(t *testing.T) {
	mat.Register(cops)
	n := 3
	xStar := mat.NewVec(n)
	xStar.AddSc(1)
	A := mat.RandN(n)
	At := A.TrView()
	AtA := mat.NewDense(n)
	AtA.Mul(At, A)

	bTmp := mat.NewVec(n)
	bTmp.Apply(A, xStar)
	b := mat.NewVec(n)
	b.Apply(At, bTmp)
	b.Scal(-2)

	c := bTmp.Nrm2Sq()
	obj, grad := opt.MakeQuadratic(AtA, b, c)
	proj := func(x mat.Vec) {
		for i := range x {
			if x[i] < 0 {
				x[i] = 0
			}
		}
	}

	solver := ProjGradSolver{
		Tol:        1e-6,
		IterMax:    5000,
		LineSearch: linesearch.InexactSolver{},
	}

	x1 := mat.NewVec(n)
	res1 := solver.Solve(obj, grad, proj, x1)
	fmt.Println(res1.Obj, res1.Iter, res1.Status)

	solver.LineSearch = linesearch.ExactSolver{Tol: 0.1}

	x2 := mat.NewVec(n)
	res2 := solver.Solve(obj, grad, proj, x2)
	fmt.Println(res2.Obj, res2.Iter, res2.Status)

	if math.Abs(res1.Obj) > 0.01 {
		t.Log(res1.Obj)
		t.Log(obj(xStar))
		t.Fail()
	}
	if math.Abs(res2.Obj) > 0.01 {
		t.Log(res2.Obj)
		t.Log(obj(xStar))
		t.Fail()
	}
}

func TestRosenbrock(t *testing.T) {
	n := 10
	scale := 10.0
	xInit := mat.NewVec(n)
	for i := 0; i < n; i++ {
		xInit[i] = rand.Float64() * scale
	}

	obj, grad := opt.MakeRosenbrock()
	proj := func(x mat.Vec) {
	}

	solver := ProjGradSolver{
		Tol:        1e-6,
		IterMax:    50000,
		LineSearch: linesearch.InexactSolver{},
	}

	x1 := mat.NewVec(n)
	x1.Copy(xInit)
	res1 := solver.Solve(obj, grad, proj, x1)
	fmt.Println(res1.Obj, res1.Iter, res1.Status)

	solver.LineSearch = linesearch.ExactSolver{Tol: 0.1}

	x2 := mat.NewVec(n)
	x2.Copy(xInit)
	res2 := solver.Solve(obj, grad, proj, x2)
	fmt.Println(res2.Obj, res2.Iter, res2.Status)
}

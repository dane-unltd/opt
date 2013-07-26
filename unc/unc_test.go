package unc

import (
	"fmt"
	"github.com/dane-unltd/linalg/mat"
	"github.com/dane-unltd/opt/linesearch"
	"math"
	"math/rand"
	"testing"
)

func TestQuadratic(t *testing.T) {
	n := 3
	xStar := mat.NewVec(n)
	xStar.AddSc(1)
	A := mat.RandN(n)
	At := A.TrView()
	AtA := mat.NewDense(n)
	AtA.Mul(At, A)

	bTmp := mat.NewVec(n)
	bTmp.Mul(A, xStar)
	b := mat.NewVec(n)
	b.Mul(At, bTmp)
	b.Scal(-2)

	c := bTmp.Nrm2Sq()
	obj, grad := MakeQuadratic(AtA, b, c)

	solver := SteepestDescentSolver{
		Tol:        1e-6,
		IterMax:    5000,
		LineSearch: linesearch.InexactSolver{},
	}

	x1 := mat.NewVec(n)
	res1 := solver.Solve(obj, grad, x1)
	fmt.Println(res1.Obj, res1.Iter, res1.Status)

	solver.LineSearch = linesearch.ExactSolver{Tol: 0.1}

	x2 := mat.NewVec(n)
	res2 := solver.Solve(obj, grad, x2)
	fmt.Println(res2.Obj, res2.Iter, res2.Status)

	solver2 := LBFGSSolver{
		Tol:        1e-6,
		IterMax:    5000,
		Mem:        5,
		LineSearch: linesearch.InexactSolver{},
	}

	x3 := mat.NewVec(n)
	res3 := solver2.Solve(obj, grad, x3)
	fmt.Println(res3.Obj, res3.Iter, res3.Status)

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
	if math.Abs(res3.Obj) > 0.01 {
		t.Log(res3.Obj)
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

	obj, grad := MakeRosenbrock()

	solver := SteepestDescentSolver{
		Tol:        1e-6,
		IterMax:    50000,
		LineSearch: linesearch.InexactSolver{},
	}

	x1 := mat.NewVec(n)
	x1.Copy(xInit)
	res1 := solver.Solve(obj, grad, x1)
	fmt.Println(res1.Obj, res1.Iter, res1.Status)

	solver.LineSearch = linesearch.ExactSolver{Tol: 0.1}

	x2 := mat.NewVec(n)
	x2.Copy(xInit)
	res2 := solver.Solve(obj, grad, x2)
	fmt.Println(res2.Obj, res2.Iter, res2.Status)

	solver2 := LBFGSSolver{
		Tol:        1e-6,
		IterMax:    5000,
		Mem:        5,
		LineSearch: linesearch.InexactSolver{},
	}

	x3 := mat.NewVec(n)
	x3.Copy(xInit)
	res3 := solver2.Solve(obj, grad, x3)
	fmt.Println(res3.Obj, res3.Iter, res3.Status)
}

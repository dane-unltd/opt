package multi

import (
	"fmt"
	"github.com/dane-unltd/linalg/clapack"
	"github.com/dane-unltd/linalg/mat"
	"github.com/dane-unltd/opt"
	"github.com/dane-unltd/opt/uni"
	"github.com/kortschak/cblas"
	"math"
	"math/rand"
	"testing"
	"time"
)

var cops struct {
	cblas.Blas
	clapack.Lapack
}

func ExampleModel(t *testing.T) {
	mat.Register(cops)

	//badly conditioned Hessian leads to zig-zagging of the steepest descent
	//algorithm
	condNo := 100.0

	A := mat.NewFromArray([]float64{condNo, 0, 0, 1}, true, 2, 2)

	//Create new 2-dimensional model.
	mdl := NewModel(2, opt.NewQuadratic(A, mat.NewVec(2), 0))

	//Register an event handler, that gets called when the model changes,
	//which is basically in every iteration of the solver.
	//Here we use the Display type to display progress in every iteration.
	mdl.AddCallback(NewDisplay(1).Update)

	//Use steepest descent solver to solve the model
	solver := NewSteepestDescent()
	solver.Solve(mdl)

	fmt.Println("x =", mdl.X)
	//should be [1,2], but because of the bad conditioning we made little
	//progress in the second dimension

	//Use a BFGS solver to refine the result:
	solver2 := NewLBFGS()
	solver2.Solve(mdl)

	fmt.Println("x =", mdl.X)
}

func TestQuadratic(t *testing.T) {
	mat.Register(cops)
	n := 10
	xStar := mat.NewVec(n)
	xStar.AddSc(1)
	A := mat.RandN(n)
	At := A.TrView()
	AtA := mat.New(n)
	AtA.Mul(At, A)

	bTmp := mat.NewVec(n)
	bTmp.Apply(A, xStar)
	b := mat.NewVec(n)
	b.Apply(At, bTmp)
	b.Scal(-2)

	c := bTmp.Nrm2Sq()

	m1 := NewModel(n, opt.NewQuadratic(AtA, b, c))
	m1.AddCallback(NewDisplay(n).Update)

	solver := NewSteepestDescent()
	solver.Solve(m1)

	t.Log(m1.ObjX, m1.Iter, m1.Status)

	solver.LineSearch = uni.NewQuadratic(false)

	m2 := NewModel(n, opt.NewQuadratic(AtA, b, c))
	m2.AddCallback(NewDisplay(n).Update)

	solver.Solve(m2)

	t.Log(m2.ObjX, m2.Iter, m2.Status)

	solver2 := NewLBFGS()

	m3 := NewModel(n, opt.NewQuadratic(AtA, b, c))
	m3.AddCallback(NewDisplay(n / 10).Update)

	solver2.Solve(m3)

	t.Log(m3.ObjX, m3.Iter, m3.Status)

	//constrained problems (constraints described as projection)
	solver3 := NewProjGrad()

	m4 := NewModel(n, opt.NewQuadratic(AtA, b, c))
	m4.Proj = opt.RealPlus{}
	m4.AddCallback(NewDisplay(n).Update)

	solver3.Solve(m4)

	t.Log(m4.ObjX, m4.Iter, m4.Status)

	if math.Abs(m1.ObjX) > 0.01 {
		t.Log(m1.ObjX)
		t.Fail()
	}
	if math.Abs(m2.ObjX) > 0.01 {
		t.Log(m2.ObjX)
		t.Fail()
	}
	if math.Abs(m3.ObjX) > 0.01 {
		t.Log(m3.ObjX)
		t.Fail()
	}
	if math.Abs(m4.ObjX) > 0.01 {
		t.Log(m4.ObjX)
		t.Fail()
	}
}

func TestRosenbrock(t *testing.T) {
	mat.Register(cops)

	n := 10
	scale := 10.0
	xInit := mat.NewVec(n)
	for i := 0; i < n; i++ {
		xInit[i] = rand.Float64() * scale
	}

	m1 := NewModel(n, opt.Rosenbrock{})
	m1.SetX(xInit, true)
	m1.AddCallback(NewDisplay(1000).Update)
	m1.Params.IterMax = 100000
	solver := NewSteepestDescent()

	solver.Solve(m1)
	t.Log(m1.ObjX, m1.Iter, m1.Status)

	solver.LineSearch = uni.NewQuadratic(false)

	m2 := NewModel(n, opt.Rosenbrock{})
	m2.SetX(xInit, true)
	m2.Params.IterMax = 100000

	//Example on how to use a callback to display information
	//Here we could plug in something more sophisticated
	m2.AddCallback(NewDisplay(1000).Update)

	//Registering a history which stores time and objective values
	hist := History{T: make([]time.Duration, 0), Obj: make([]float64, 0)}
	m2.AddCallback(hist.Update)

	solver.Solve(m2)
	t.Log(m2.ObjX, m2.Iter, m2.Status)

	solver2 := NewLBFGS()

	m3 := NewModel(n, opt.Rosenbrock{})
	m3.SetX(xInit, true)
	m3.AddCallback(NewDisplay(10).Update)
	solver2.Solve(m3)
	t.Log(m3.ObjX, m3.Iter, m3.Status)

	m4 := NewModel(n, opt.Rosenbrock{})
	m4.SetX(xInit, true)
	m4.AddCallback(NewDisplay(10).Update)
	solver2.LineSearch = uni.NewQuadratic(false)
	solver2.Solve(m4)
	t.Log(m4.ObjX, m4.Iter, m4.Status)

	if math.Abs(m1.ObjX) > 0.1 {
		t.Log(m1.ObjX)
		t.Fail()
	}
	if math.Abs(m2.ObjX) > 0.1 {
		t.Log(m2.ObjX)
		t.Fail()
	}
	if math.Abs(m3.ObjX) > 0.1 {
		t.Log(m3.ObjX)
		t.Fail()
	}
	if math.Abs(m4.ObjX) > 0.1 {
		t.Log(m4.ObjX)
		t.Fail()
	}
}

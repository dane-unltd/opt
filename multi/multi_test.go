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

func init() {
	mat.Register(cops)
}

func TestExampleModel(t *testing.T) {
	//badly conditioned Hessian leads to zig-zagging of the steepest descent
	//algorithm
	condNo := 100.0
	optSol := mat.Vec{1, 2}

	A := mat.NewFromArray([]float64{condNo, 0, 0, 1}, true, 2, 2)
	b := mat.Vec{-2 * optSol[0] * condNo, -2 * optSol[1]}
	c := -0.5 * mat.Dot(b, optSol)

	//Create new 2-dimensional model.
	mdl := NewModel(2, opt.NewQuadratic(A, b, c))

	//Register an event handler, that gets called when the model changes,
	//which is basically in every iteration of the solver.
	//Here we use the Display type to display progress in every iteration.
	mdl.AddCallback(NewDisplay(1))

	mdl.Params.IterMax = 5

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
	m1.AddCallback(NewDisplay(n))

	solver := NewSteepestDescent()
	solver.Solve(m1)

	t.Log(m1.ObjX, m1.Iter, m1.Status)

	solver.LineSearch = uni.NewQuadratic(false)

	m2 := NewModel(n, opt.NewQuadratic(AtA, b, c))
	m2.AddCallback(NewDisplay(n))

	solver.Solve(m2)

	t.Log(m2.ObjX, m2.Iter, m2.Status)

	solver2 := NewLBFGS()

	m3 := NewModel(n, opt.NewQuadratic(AtA, b, c))
	m3.AddCallback(NewDisplay(n / 10))

	solver2.Solve(m3)

	t.Log(m3.ObjX, m3.Iter, m3.Status)

	//constrained problems (constraints described as projection)
	solver3 := NewProjGrad()

	m4 := NewModel(n, opt.NewQuadratic(AtA, b, c))
	m4.Proj = opt.RealPlus{}
	m4.AddCallback(NewDisplay(n))

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
	m1.AddCallback(NewDisplay(1000))
	m1.Params.IterMax = 100000
	m1.Params.FunEvalMax = 100000
	solver := NewSteepestDescent()

	solver.Solve(m1)
	t.Log(m1.ObjX, m1.Iter, m1.Status)

	solver.LineSearch = uni.NewQuadratic(false)

	m2 := NewModel(n, opt.Rosenbrock{})
	m2.SetX(xInit, true)
	m2.Params.IterMax = 100000
	m2.Params.FunEvalMax = 100000

	//Example on how to use a callback to display information
	//Here we could plug in something more sophisticated
	m2.AddCallback(NewDisplay(1000))

	//Registering a history which stores time and objective values
	hist := &History{T: make([]time.Duration, 0), Obj: make([]float64, 0)}
	m2.AddCallback(hist)

	solver.Solve(m2)
	t.Log(m2.ObjX, m2.Iter, m2.Status)

	solver2 := NewLBFGS()

	m3 := NewModel(n, opt.Rosenbrock{})
	m3.SetX(xInit, true)
	m3.AddCallback(NewDisplay(10))
	solver2.Solve(m3)
	t.Log(m3.ObjX, m3.Iter, m3.Status)

	m4 := NewModel(n, opt.Rosenbrock{})
	m4.SetX(xInit, true)
	m4.AddCallback(NewDisplay(10))
	solver2.LineSearch = uni.NewQuadratic(false)
	solver2.Solve(m4)
	t.Log(m4.ObjX, m4.Iter, m4.Status)

	m5 := NewModel(n, opt.Rosenbrock{})
	m5.SetX(xInit, true)
	m5.AddCallback(NewDisplay(10))
	m5.Params.IterMax = 1000
	solver2.LineSearch = uni.NewCubic()
	solver2.Solve(m5)
	t.Log(m5.ObjX, m5.Iter, m5.Status)

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
	if math.Abs(m5.ObjX) > 0.1 {
		t.Log(m5.ObjX)
		t.Fail()
	}
}

type rb struct {
	opt.Rosenbrock
}

func (r rb) ValGrad() {}

type rosTest struct {
}

func (r rosTest) Val(x mat.Vec) float64 {
	return math.Pow(x[0]-2, 4) + math.Pow(x[0]-2*x[1], 2)
}

func TestSolve(t *testing.T) {
	mat.Register(cops)

	xInit := mat.RandVec(10).Scal(10.0)

	mdl := Solve(opt.Rosenbrock{}, xInit, nil, NewDisplay(10))

	t.Log(mdl.Status, mdl.ObjX, mdl.Iter)
	if math.Abs(mdl.ObjX) > 0.1 {
		t.Fail()
	}

	params := NewParams()
	params.IterMax = 100000

	mdl = SolveGradProjected(opt.Rosenbrock{}, opt.RealPlus{}, xInit,
		params, NewDisplay(1000))
	t.Log(mdl.Status, mdl.ObjX, mdl.Iter)

	params.XTolAbs = 1e-9
	params.XTolRel = 0
	params.FunTolRel = 0
	params.FunTolAbs = 0
	params.FunEvalMax = 100000
	mdl = Solve(rb{}, xInit, params, NewDisplay(1))
	t.Log(mdl.Status, mdl.ObjX, mdl.Iter, mdl.X)

	xInit = mat.Vec{0, 3}
	mdl = Solve(rosTest{}, xInit, params, NewDisplay(1))
	t.Log(mdl.Status, mdl.ObjX, mdl.Iter)
}

package multi

import (
	"fmt"
	"github.com/dane-unltd/linalg/clapack"
	"github.com/dane-unltd/linalg/mat"
	"github.com/dane-unltd/opt"
	"github.com/dane-unltd/opt/uni"
	"github.com/kortschak/cblas"
	"math"
	"testing"
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

	//define objective function
	fun := opt.NewQuadratic(A, b, c)

	//set inital solution estimate
	sol := NewSolution(mat.NewVec(2))

	//set termination parameters
	p := NewParams()
	p.IterMax = 5

	//Use steepest descent solver to solve the model
	result := NewSteepestDescent().Solve(fun, sol, p, NewDisplay(1))

	fmt.Println("x =", result.X)
	//should be [1,2], but because of the bad conditioning we made little
	//progress in the second dimension

	//Use a BFGS solver to refine the result:
	result = NewLBFGS().Solve(fun, result.Solution, p, NewDisplay(1))

	fmt.Println("x =", result.X)
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

	//Define input arguments
	obj := opt.NewQuadratic(AtA, b, c)
	p := NewParams()
	sol := NewSolution(mat.NewVec(n))

	//Steepest descent with armijo
	stDesc := NewSteepestDescent()
	res1 := stDesc.Solve(obj, sol, p, NewDisplay(100))

	t.Log(res1.ObjX, res1.FunEvals, res1.GradEvals, res1.Status)

	//Steepest descent with Quadratic
	stDesc.LineSearch = uni.DerivWrapper{uni.NewQuadratic()}
	res2 := stDesc.Solve(obj, sol, p, NewDisplay(100))

	t.Log(res2.ObjX, res2.FunEvals, res2.GradEvals, res2.Status)

	//LBFGS with armijo
	lbfgs := NewLBFGS()
	res3 := lbfgs.Solve(obj, sol, p, NewDisplay(10))

	t.Log(res3.ObjX, res3.FunEvals, res3.GradEvals, res3.Status)

	//constrained problems (constraints described as projection)
	projGrad := NewProjGrad()

	res4 := projGrad.Solve(obj, opt.RealPlus{}, sol, p, NewDisplay(100))

	t.Log(res4.ObjX, res4.FunEvals, res4.GradEvals, res4.Status)

	if math.Abs(res1.ObjX) > 0.01 {
		t.Fail()
	}
	if math.Abs(res2.ObjX) > 0.01 {
		t.Fail()
	}
	if math.Abs(res3.ObjX) > 0.01 {
		t.Fail()
	}
	if math.Abs(res4.ObjX) > 0.01 {
		t.Fail()
	}
}

func TestRosenbrock(t *testing.T) {
	mat.Register(cops)

	n := 10
	scale := 10.0
	xInit := mat.RandVec(n).Scal(scale)

	//Define input arguments
	obj := opt.Rosenbrock{}
	p := NewParams()
	p.FunEvalMax = 100000
	p.IterMax = 100000
	sol := NewSolution(xInit)

	//Steepest descent with armijo
	stDesc := NewSteepestDescent()
	res1 := stDesc.Solve(obj, sol, p, NewDisplay(100))

	t.Log(res1.ObjX, res1.FunEvals, res1.GradEvals, res1.Status)

	//Steepest descent with Quadratic
	stDesc.LineSearch = uni.DerivWrapper{uni.NewQuadratic()}
	res2 := stDesc.Solve(obj, sol, p, NewDisplay(100))

	t.Log(res2.ObjX, res2.FunEvals, res2.GradEvals, res2.Status)

	//LBFGS with armijo
	lbfgs := NewLBFGS()
	res3 := lbfgs.Solve(obj, sol, p, NewDisplay(10))

	t.Log(res3.ObjX, res3.FunEvals, res3.GradEvals, res3.Status)

	//LBFGS with Quadratic
	lbfgs.LineSearch = uni.DerivWrapper{uni.NewQuadratic()}
	res4 := lbfgs.Solve(obj, sol, p, NewDisplay(10))

	t.Log(res4.ObjX, res4.FunEvals, res4.GradEvals, res4.Status)

	//LBFGS with Cubic
	lbfgs.LineSearch = uni.NewCubic()
	res5 := lbfgs.Solve(obj, sol, p, NewDisplay(10))

	t.Log(res5.ObjX, res5.FunEvals, res5.GradEvals, res5.Status)

	if math.Abs(res1.ObjX) > 0.01 {
		t.Fail()
	}
	if math.Abs(res2.ObjX) > 0.01 {
		t.Fail()
	}
	if math.Abs(res3.ObjX) > 0.01 {
		t.Fail()
	}
	if math.Abs(res4.ObjX) > 0.01 {
		t.Fail()
	}
	if math.Abs(res5.ObjX) > 0.01 {
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
	sol := NewSolution(xInit)

	result := Solve(opt.Rosenbrock{}, sol, nil, NewDisplay(10))

	t.Log(result.Status, result.ObjX, result.Iter)
	if math.Abs(result.ObjX) > 0.1 {
		t.Fail()
	}

	params := NewParams()
	params.IterMax = 100000

	result = SolveGradProjected(opt.Rosenbrock{}, opt.RealPlus{}, sol,
		params, NewDisplay(1000))
	t.Log(result.Status, result.ObjX, result.Iter)

	params.XTolAbs = 1e-9
	params.XTolRel = 0
	params.FunTolRel = 0
	params.FunTolAbs = 0
	params.FunEvalMax = 100000
	result = Solve(rb{}, sol, params, NewDisplay(1))
	t.Log(result.Status, result.ObjX, result.Iter, result.X)

	xInit = mat.Vec{0, 3}
	sol.SetX(xInit, false)
	result = Solve(rosTest{}, sol, params, NewDisplay(1))
	t.Log(result.Status, result.ObjX, result.Iter)
}

package multi

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/dane-unltd/opt"
	"github.com/gonum/blas"
	"github.com/gonum/blas/cblas"
	"github.com/gonum/blas/dbw"
)

var cops struct {
	cblas.Blas
}

func init() {
	dbw.Register(cops)
}

func TestExampleModel(t *testing.T) {
	//badly conditioned Hessian leads to zig-zagging of the steepest descent
	//algorithm
	condNo := 100.0
	optSol := dbw.NewVector([]float64{1, 2})

	A := dbw.NewGeneral(2, 2, []float64{condNo, 0, 0, 1})
	b := dbw.NewVector([]float64{-2 * optSol.Data[0] * condNo,
		-2 * optSol.Data[1]})
	c := -0.5 * dbw.Dot(b, optSol)

	//define objective function
	fun := opt.NewQuadratic(A, b.Data, c)

	//set inital solution estimate
	sol := NewSolution(make([]float64, 2))

	//set termination parameters
	p := NewParams()
	p.IterMax = 5

	//Use steepest descent solver to solve the model
	solver := NewSearchBased(SteepestDescent{}, Backtracking{Armijo: 0.2})
	solver.Optimize(fun, sol, p, GradConv{1e-3}, NewDisplay(1))

	fmt.Println("x =", sol.X)
	//should be [1,2], but because of the bad conditioning we made little
	//progress in the second dimension

	//Use a BFGS solver to refine the result:
	solver = NewSearchBased(new(LBFGS), Backtracking{Armijo: 0.2})
	solver.Optimize(fun, sol, p, GradConv{1e-3}, NewDisplay(1))

	fmt.Println("x =", sol.X)
}

func TestQuadratic(t *testing.T) {
	dbw.Register(cops)

	n := 5

	xStar := dbw.NewVector(make([]float64, n))
	for i := range xStar.Data {
		xStar.Data[i] = 1
	}
	A := dbw.NewGeneral(n, n, make([]float64, n*n))
	for i := range A.Data {
		A.Data[i] = rand.NormFloat64()
	}
	AtA := dbw.NewGeneral(n, n, make([]float64, n*n))
	dbw.Gemm(blas.Trans, blas.NoTrans, 1, A, A, 0, AtA)

	bTmp := dbw.NewVector(make([]float64, n))
	dbw.Gemv(blas.NoTrans, 1, A, xStar, 0, bTmp)
	b := dbw.NewVector(make([]float64, n))
	dbw.Gemv(blas.Trans, -2, A, bTmp, 0, b)

	c := dbw.Dot(bTmp, bTmp)

	//Define input arguments
	obj := opt.NewQuadratic(AtA, b.Data, c)

	//Steepest descent with Backtracking
	sol := NewSolution(make([]float64, n))
	stDesc := NewSearchBased(SteepestDescent{}, Backtracking{Armijo: 0.2})
	status := stDesc.Optimize(obj, sol, GradConv{1e-6}, NewDisplay(50))

	t.Log(sol.Obj, stDesc.stats.FunEvals, stDesc.stats.GradEvals, status)
	if math.Abs(sol.Obj) > 0.01 {
		t.Fail()
	}

	//LBFGS with Backtracking
	sol = NewSolution(make([]float64, n))
	lbfgs := NewSearchBased(new(LBFGS), Backtracking{Armijo: 0.2})
	status = lbfgs.Optimize(obj, sol, GradConv{1e-6}, NewDisplay(1))

	t.Log(sol.Obj, lbfgs.stats.FunEvals, lbfgs.stats.GradEvals, status)
	if math.Abs(sol.Obj) > 0.01 {
		t.Fail()
	}
}

func copyOf(x []float64) []float64 {
	r := make([]float64, len(x))
	copy(r, x)
	return r
}

func TestRosenbrock(t *testing.T) {
	dbw.Register(cops)

	n := 10
	scale := 10.0

	xInit := make([]float64, n)
	for i := range xInit {
		xInit[i] = scale * rand.Float64()
	}

	//Define input arguments
	obj := opt.Rosenbrock{}

	//Steepest descent with Backtracking
	sol := NewSolution(copyOf(xInit))
	stDesc := NewSearchBased(SteepestDescent{}, Backtracking{Armijo: 0.2})
	status := stDesc.Optimize(obj, sol, GradConv{1e-6},
		NewDisplay(500), Termination{5000, 10 * time.Second})

	t.Log(sol.Obj, stDesc.stats.FunEvals, stDesc.stats.GradEvals, status)

	//LBFGS with Backtracking
	sol = NewSolution(copyOf(xInit))
	lbfgs := NewSearchBased(new(LBFGS), Backtracking{Armijo: 0.2})
	status = lbfgs.Optimize(obj, sol, GradConv{1e-6}, NewDisplay(10))

	t.Log(sol.Obj, lbfgs.stats.FunEvals, lbfgs.stats.GradEvals, status)
}

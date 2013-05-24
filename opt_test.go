package opt

import (
	"fmt"
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
	A := matrix.NewFromArray([]float64{1, 1}, false, 1, 2)
	b := matrix.Vec{1}
	c := matrix.Vec{1, 0}

	x := linprog(c, A, b)
	fmt.Println(x)
}

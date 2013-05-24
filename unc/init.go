package unc

import (
	"github.com/dane-unltd/linalg/clapack"
	"github.com/dane-unltd/linalg/matrix"
	"github.com/kortschak/cblas"
)

type matops struct {
	cblas.Blas
	clapack.Lapack
}

func init() {
	matrix.Register(matops{})
}

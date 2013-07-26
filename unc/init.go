package unc

import (
	"github.com/dane-unltd/linalg/clapack"
	"github.com/dane-unltd/linalg/mat"
	"github.com/kortschak/cblas"
)

type matops struct {
	cblas.Blas
	clapack.Lapack
}

func init() {
	mat.Register(matops{})
}

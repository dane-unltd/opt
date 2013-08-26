package uni

type F interface {
	F(float64) float64
}

type FdF interface {
	F
	DF(float64) float64
	FdF(float64) (float64, float64)
}

type FdFddF interface {
	FdF
	DDF(float64) float64
	FdFddF(float64) (float64, float64, float64)
}

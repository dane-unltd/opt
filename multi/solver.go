package multi

type Solver interface {
	Solve(m *Model) error
}

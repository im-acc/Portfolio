
# Jacobi method for solving 2D laplace equation by finite diffences
# on [0,1] x [0,1] square with v(B)=0 {B : boundary}
function laplace_solver(f, n=200, niter=50)
    v = zeros(n+2,n+2)
    h = 1/n
    for _ in 1:niter
        for i in 2:n+1, j in 2:n+1
            v[i,j] = ( v[i-1,j] + v[i,j-1] + v[i+1,j] + v[i,j+1] - h^2 * f(i*h, j*h) ) / 4
        end
    end
    return v
end


function f(x,y)
    return sin(pi*x)*sin(pi*y)/pi
end


@time begin
sol = laplace_solver(f)
end

using Plots
heatmap(sol)

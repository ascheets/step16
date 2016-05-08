( ((pn[1:-1,1]+pn[1:-1,-1])*dy**2+(pn[2:,0]+pn[0:-2,0])*dx**2)/(2*(dx**2+dy**2)) - #remnants of laplacian pressure business
                        rho*dx**2*dy**2/(2*(dx**2+dy**2))* #weird term that comes about from transposing
                        b[1:-1,0] ) #source term



( ((pn[1:-1,0]+pn[1:-1,-2])*dy**2+(pn[2:,-1]+pn[0:-2,-1])*dx**2)/(2*(dx**2+dy**2)) - #remnants of laplacian pressure business
                        rho*dx**2*dy**2/(2*(dx**2+dy**2))* #weird term that comes about from transposing
                        b[1:-1,-1] ) #source term

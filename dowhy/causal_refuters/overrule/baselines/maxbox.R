#####################
# maxbox.R
#####################
# INPUT:
#####################
# X: matrix of p >= 1 covariates to be used to define the maximal box
# exclude: indicator of whether an individual must be excluded from the resulting subpopulation
# cutoff: At which point should the branch and bound switch from a depth first to best first strategy. Can be helpful to make this nonzero in order to get an initial feasible solution
# max.iter: maximum number of iterations. If this is exceeded, we instead use the best solution found up until that point (even though it may have suboptimal cardinality)
######################
# OUTPUT:
######################
# BOUNDS: px2 matrix of upper and lower boundaries of maximal box
# cardinality: number of individuals in resulting subpopulation
# index.subpop: binary indicators for whether each row of X is included in the subpopulation
#####################

maxbox = function(X, exclude, cutoff = 20, max.iter = 10000)
{
  if(is.vector(X))
  {
    X = matrix(X, length(X),1)
  }
  exclude = 1*exclude
  if(any(exclude!=0 & exclude !=1)|| nrow(X) != length(exclude))
  {
    stop("exclude must be a binary vector of length nrow(X)")
  }
  exclude = (exclude == 1)
  B = X[!exclude,,drop=F]
  R = X[exclude,,drop=F]
  red.outside = function(R, L, U)
  {
    n = nrow(R)
    d = ncol(R) 
    LMAT = matrix(L, n, d, byrow = T)
    UMAT = matrix(U, n, d, byrow = T)
    f = rowSums(R < LMAT | R > UMAT) != 0
    f
  }
  
  card.blue = function(B, L, U)
  {
    n = nrow(B)
    d = ncol(B) 
    LMAT = matrix(L, n, d, byrow = T)
    UMAT = matrix(U, n, d, byrow = T)
    which.in = 1 - rowSums(B < LMAT | B > UMAT) == 1
    which.in
  }
  
  upper.bound = function(B, R, L, U)
  {
    out = red.outside(R, L, U)
    f = all(out)
    ind = which(out == F)
    d = ncol(R)
    ind.blue = card.blue(B, L, U)
    split1 = 0
    
    if(f == T)
    {
      UP = sum(ind.blue)
    }
    if(f == F)
    {
      len = length(ind)
      up = rep(0, len)
      for(i in 1:len)
      {
        for(j in 1:d)
        {
          
          up[i] = max(up[i], sum(ind.blue*(B[,j] < R[ind[i],j])), sum(ind.blue*(B[,j] > R[ind[i],j])))
        }
      }	
      
      UP = min(up[up!=0])
      split1 = ind[which(up == min(up))[1]]
    }
    return(list(UP = UP, split = split1))
  }
  
  
  
  feasible = function(B, R, L.under, L.over, U.under, U.over)
  {
    ratio = 0
    FEAS = F
    cond = rep(F, 5)
    n.b = nrow(B)
    split1 = 0
    d = ncol(B) 
    LMAT.B = matrix(L.under, n.b, d, byrow = T)
    UMAT.B = matrix(U.over, n.b, d, byrow = T)
    in.box = (1 - rowSums(B < LMAT.B | B > UMAT.B) == 1)
    cblue = sum(in.box)
    cond[1] = cblue > 0
    cond[2] = all(L.under <= L.over)
    cond[3] = all(U.under <= U.over)
    cond[4] = all(L.under <= U.over)
    n.r = nrow(R)
    LMAT.R = matrix(L.over, n.r, d, byrow = T)
    UMAT.R = matrix(U.under, n.r, d, byrow = T)
    cond[5] = sum(1 - rowSums(R < LMAT.R | R > UMAT.R) == 1) == 0
    max.U = 0
    if(all(cond))
    {
      FEAS = T
      card.R = sum(1 - red.outside(R,L.under, U.over))
      temp = upper.bound(B, R, L.under, U.over)
      max.U = temp$UP
      split1 = temp$split
      ratio = max.U/card.R
    }
    return(list(feasible = FEAS, ratio = ratio, split = split1, num.blue = cblue, blue.in = in.box, UB = max.U))			
  }
  
  BLUE = B
  RED = R
  
  d = ncol(RED)
  n.r = nrow(RED)
  n.n = nrow(BLUE)
  L.under = apply(BLUE, 2, min)
  U.over = apply(BLUE, 2, max)
  L.over = U.over
  U.under = L.under
  satisfy = F
  L.current = L.under
  U.current = U.over
  L.undernode = L.under
  L.overnode = L.over
  U.undernode = U.under
  U.overnode = U.over
  feas = feasible(B, R, L.under, L.over, U.under, U.over)
  blue.in = feas$blue.in
  card.current = feas$num.blue
  lev = 1
  split1 = feas$split
  split.q = c()
  ratio.q = c()
  Lunder.q = c()
  Lover.q = c()
  Uunder.q = c()
  Uover.q = c()
  UB.q = c()
  current.max = 0
  stop = F
  niter = 1
  queue = c()
  lev.q = c()
  L.underprop = matrix(0, d, 2*d)
  L.overprop = matrix(0, d, d)
  U.underprop = matrix(0, d, d)
  U.overprop = matrix(0, d, 2*d)
  sum.r = 10
  L.underprop = matrix(0, d, 2*d)
    
  U.overprop = matrix(0, d, 2*d)
  ptm = proc.time()
  while(stop == F)
  {		
    if(sum.r > 0)
    {
      a = RED[split1,,drop = F]
      
      empty = rep(T, 2*d)
      L.overprop = matrix(0, d, d)
      U.underprop = matrix(0, d, d)
      
      for(i in 1:d)
      {
        if(sum(blue.in*(BLUE[,i] > a[i]))> 1)
        {
          L.underprop[,i] = apply(BLUE[(blue.in*(BLUE[,i] > a[i])==1),,drop = F],2, min)
          U.overprop[,i] = apply(BLUE[(blue.in*(BLUE[,i] > a[i])==1),,drop = F],2, max)
          empty[i] = F
        }
        if(sum(blue.in*(BLUE[,i] < a[i])) > 1)
        {
          L.underprop[,i+d] = apply(BLUE[(blue.in*(BLUE[,i] < a[i])==1),, drop = F],2, min)
          U.overprop[,i+d] = apply(BLUE[(blue.in*(BLUE[,i] < a[i])==1),, drop = F],2, max)
          empty[i+d] = F
        }
        for(j in 1:d)
        {
          if(j < i)
          {
            L.overprop[j,i] = min(L.overnode[j], a[j])
            U.underprop[j,i] = max(U.undernode[j], a[j])
          }
          if(j >= i)
          {
            L.overprop[j,i] = L.overnode[j]
            U.underprop[j,i] = U.undernode[j]
          }
        }
        
        
      }
      L.overprop = cbind(L.overprop, L.overprop)
      U.underprop = cbind(U.underprop, U.underprop)
      feasprop = rep(F, 2*d)
      ratioprop = rep(NA, 2*d)
      splitprop = rep(0, 2*d)
      UBprop = rep(0, 2*d)
      levprop = rep(0, 2*d)
      temp = 0
      for(i in (1:(2*d))[!empty])
      {
        temp = feasible(B, R, L.underprop[,i], L.overprop[,i], U.underprop[,i], U.overprop[,i])
        feasprop[i] = temp$feasible
        ratioprop[i] = temp$ratio
        splitprop[i] = as.numeric(temp$split)	
        UBprop[i] = temp$UB
        levprop[i] = lev + 1
        
      }
      
      
      
      split.q = c(split.q, splitprop[feasprop])
      lev.q = c(lev.q, levprop[feasprop])
      if(d > 1)
      {
        Uover.q = cbind(Uover.q, U.overprop[,feasprop, drop = F])
        Uunder.q = cbind(Uunder.q, U.underprop[,feasprop, drop = F])
        Lover.q = cbind(Lover.q, L.overprop[,feasprop, drop = F])
        Lunder.q = cbind(Lunder.q, L.underprop[,feasprop, drop = F])
      }
      if(d == 1)
      {
        lll = length(c(Uover.q, U.overprop))
        Uover.q = matrix(c(Uover.q, U.overprop),1, lll) 
        Uunder.q = matrix(c(Uunder.q, U.underprop),1, lll)
        Lover.q =  matrix(c(Lover.q, L.overprop),1,lll)
        Lunder.q = matrix(c(Lunder.q, L.underprop), 1, lll)
      }
      
      ratio.q = c(ratio.q, ratioprop[feasprop])
      UB.q = c(UB.q, UBprop[feasprop])
      
      if(niter > cutoff)
      {
        order.new =order(ratio.q, decreasing = T)
      }
      else
      {
        order.new = order(UB.q, decreasing = T)
        
      }
      
      
      split.q = split.q[order.new]
      lev.q = lev.q[order.new]
      Lunder.q = Lunder.q[,order.new, drop = F]
      Lover.q = Lover.q[,order.new, drop = F]
      Uunder.q = Uunder.q[,order.new, drop = F]
      Uover.q = Uover.q[,order.new, drop = F]
      ratio.q = ratio.q[order.new]
      UB.q = UB.q[order.new]
      
    }
    ubmax = max(UB.q)
    noprune = (UB.q > current.max)
    niter = niter + 1
    if(length(noprune) > 1)
    {
      split.q = split.q[noprune]
      lev.q = lev.q[noprune]
      Lunder.q = Lunder.q[,noprune, drop = F]
      Lover.q = Lover.q[,noprune, drop = F]
      Uunder.q = Uunder.q[,noprune, drop = F]
      Uover.q = Uover.q[,noprune, drop = F]
      ratio.q = ratio.q[noprune, drop = F]
      UB.q = UB.q[noprune]
      ls = length(split.q)
    }
    if(length(noprune) == 1 & sum(noprune) == 1)
    {
      ls = 1
    }
    if(sum(noprune)==0)
    {
      ls = 0
      stop = T
    }
    
    if(ls > 1)
    {
      split1 = split.q[1]
      lev = lev.q[1]
      L.overnode = Lover.q[,1]
      L.undernode = Lunder.q[,1]
      U.overnode = Uover.q[,1]
      U.undernode = Uunder.q[,1]
      split.q = split.q[-1]
      lev.q = lev.q[-1]
      Lunder.q = Lunder.q[,-1, drop = F]
      Lover.q = Lover.q[,-1, drop = F]
      Uunder.q = Uunder.q[,-1, drop = F]
      Uover.q = Uover.q[,-1, drop = F]
      ratio.q = ratio.q[-1]
      UB.q = UB.q[-1]
    }
    else if (ls == 1)
    {
      split1 = split.q
      lev = lev.q
      L.overnode = Lover.q
      L.undernode = Lunder.q
      U.overnode = Uover.q
      U.undernode = Uunder.q
      split.q = c()
      Lunder.q = c()
      Lover.q = c()
      Uunder.q = c()
      Uover.q = c()
      ratio.q = c()
      lev.q = c()
      UB.q = c()
    }
    else
    {
      stop = T
    }
    
    blue.in = feasible(B, R, L.undernode, L.overnode, U.undernode, U.overnode)$blue.in
    sum.r = sum(1-red.outside(R, L.undernode, U.overnode))
    if(sum.r == 0)
    {
      if(sum(blue.in) > current.max)
      {
        current.max = sum(blue.in)
        L.max = L.undernode
        U.max = U.overnode
        mub = 0
        if(length(UB.q) >= 1)
        {
          mub = max(UB.q)
        }
        if(current.max >= mub)
        {
          stop = T
        }
      }
      if(is.null(split.q))
      {
        stop = T
      }
    }
    
    if(niter >= max.iter)
    {
      
      if(current.max == 0)
      {
        
        stop("Iteration limit reached with no feasible solution found")
      }
      stop = T
    }
    #print(c(niter, current.max, length(split.q), ubmax))
  }
  Btemp = cbind(L.max, U.max)
  rownames(Btemp) = colnames(X)
  return(list(BOUNDS = Btemp, cardinality = current.max, index.subpop =  card.blue(X, L.max, U.max)))
}
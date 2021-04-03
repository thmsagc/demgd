*solução*. Você tem duas coisas a mostrar:

1. Se $\varphi(a) = b^k$ é um homomorfismo, então $t$ divide $sk$. Isso você fez bem, pois basta observar que 
   $$
   \nonumber
   e' = \varphi(e) = \varphi(a^s) = \varphi(a)^s = (b^k)^s = b^{sk}.
   $$
   Como $t$ é a ordem de $G'$, então precisamos ter $t\mid sk$.

2. Para a recíproca a hipótese é que $t\mid sk$ e, a partir daí, você precisa encontrar um homomorfismo de grupos $\varphi$ tal que $\varphi(a) = b^k$.  Para tal defina
   $$
   \nonumber
   \begin{array}{ccl}
   G &\to& G' \\
   a^j & \mapsto & \varphi(a^j) = b^{jk}.
   \end{array}
   $$
   É preciso mostrar que essa aplicação está bem definida, pois uma vez que os grupos tem ordens finitas, diferentes potências de $a$ representam  o mesmo elemento. Assim sejam $n_1>n_2$ tais que 
   $$
   \nonumber
   a^{n_1} = a^{n_2} \quad\text{como elementos de}\quad G.
   $$
    Precisamos mostrar que $\varphi ( a^{n_1} ) = \varphi (a^{n_2})$.  De $a^{n_1} = a^{n_2}$, obtemos $a^{n_1-n_2} = e$, assim obtemos que $s\mid n_1-n_2$; isto é, podemos escrever $n_1-n_2 = q_1s$, para algum $q_1\in\mathbb{Z}$.    Por outro lado, segue da definição de $\varphi$ que 
   $$
   \nonumber
   \varphi (a^{n_1 - n_2}) = b^{k(n_1-n_2)} = b^{kq_1s}= b^{skq_1}.
   $$
   Por hipótese $t\mid sk$, assim podemos escrever $sk = q_2t$ para algum $q_2\in\mathbb{Z}$, substituindo obtemos
   $$
   \nonumber
   \varphi(a^{n_1-n_2}) = b^{k(n_1-n_2)}=b^{skq_1} =b^{q_2tq_2} = (b^{t})^{q_1a_2} = (e')^{q_1a_2} = e 
   $$
   Isso mostra que 
   $$
   \nonumber
   b^{kn_1}b^{kn_2} = e' \Rightarrow b^{kn_1} = b^{kn_2} \Rightarrow \varphi(a^{n_1}) = \varphi(a^{n_2}),
   $$
   portanto $\varphi$ está bem definida.  Finalmente, segue diretamente da definição de $\varphi$ é um homomorfismo de grupos.  De fato, suponha que $x,y\in G$, então $x = a^m$ e $y = a^n$, então 
   $$
   \nonumber
   \varphi(xy) = \varphi(a^{m+n}) = b^{k(m+n)}=b^{mk}\cdot b^{nk} = \varphi(a^n)\cdot \varphi(a^m) = \varphi (x)\cdot \varphi(y).
   $$
   Note ainda que $\varphi (a) = b^k$, como queríamos.


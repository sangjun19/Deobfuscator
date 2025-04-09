	.file	"GabzinhoHD_Mercado-C_Mercado_flatten.c"
	.text
	.globl	_TIG_IZ_q5CY_argc
	.bss
	.align 4
	.type	_TIG_IZ_q5CY_argc, @object
	.size	_TIG_IZ_q5CY_argc, 4
_TIG_IZ_q5CY_argc:
	.zero	4
	.globl	total_produtos
	.align 4
	.type	total_produtos, @object
	.size	total_produtos, 4
total_produtos:
	.zero	4
	.globl	carrinho
	.align 32
	.type	carrinho, @object
	.size	carrinho, 440
carrinho:
	.zero	440
	.globl	_TIG_IZ_q5CY_argv
	.align 8
	.type	_TIG_IZ_q5CY_argv, @object
	.size	_TIG_IZ_q5CY_argv, 8
_TIG_IZ_q5CY_argv:
	.zero	8
	.globl	estoque
	.align 32
	.type	estoque, @object
	.size	estoque, 400
estoque:
	.zero	400
	.globl	total_itens_carrinho
	.align 4
	.type	total_itens_carrinho, @object
	.size	total_itens_carrinho, 4
total_itens_carrinho:
	.zero	4
	.globl	_TIG_IZ_q5CY_envp
	.align 8
	.type	_TIG_IZ_q5CY_envp, @object
	.size	_TIG_IZ_q5CY_envp, 8
_TIG_IZ_q5CY_envp:
	.zero	8
	.section	.rodata
.LC1:
	.string	"Saindo..."
.LC2:
	.string	"1. Cadastrar Produto"
.LC3:
	.string	"2. Listar Produtos"
.LC4:
	.string	"3. Comprar Produto"
.LC5:
	.string	"4. Visualizar Carrinho"
.LC6:
	.string	"5. Sair"
.LC7:
	.string	"Escolha uma opcao: "
.LC8:
	.string	"%d"
.LC9:
	.string	"Opcao invalida!"
	.text
	.globl	main
	.type	main, @function
main:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movl	$0, total_itens_carrinho(%rip)
	nop
.L2:
	movl	$0, total_produtos(%rip)
	nop
.L3:
	movl	$0, carrinho(%rip)
	movb	$0, 4+carrinho(%rip)
	movb	$0, 5+carrinho(%rip)
	movb	$0, 6+carrinho(%rip)
	movb	$0, 7+carrinho(%rip)
	movb	$0, 8+carrinho(%rip)
	movb	$0, 9+carrinho(%rip)
	movb	$0, 10+carrinho(%rip)
	movb	$0, 11+carrinho(%rip)
	movb	$0, 12+carrinho(%rip)
	movb	$0, 13+carrinho(%rip)
	movb	$0, 14+carrinho(%rip)
	movb	$0, 15+carrinho(%rip)
	movb	$0, 16+carrinho(%rip)
	movb	$0, 17+carrinho(%rip)
	movb	$0, 18+carrinho(%rip)
	movb	$0, 19+carrinho(%rip)
	movb	$0, 20+carrinho(%rip)
	movb	$0, 21+carrinho(%rip)
	movb	$0, 22+carrinho(%rip)
	movb	$0, 23+carrinho(%rip)
	movb	$0, 24+carrinho(%rip)
	movb	$0, 25+carrinho(%rip)
	movb	$0, 26+carrinho(%rip)
	movb	$0, 27+carrinho(%rip)
	movb	$0, 28+carrinho(%rip)
	movb	$0, 29+carrinho(%rip)
	movb	$0, 30+carrinho(%rip)
	movb	$0, 31+carrinho(%rip)
	movb	$0, 32+carrinho(%rip)
	movb	$0, 33+carrinho(%rip)
	pxor	%xmm0, %xmm0
	movss	%xmm0, 36+carrinho(%rip)
	movl	$0, 40+carrinho(%rip)
	movl	$0, 44+carrinho(%rip)
	movb	$0, 48+carrinho(%rip)
	movb	$0, 49+carrinho(%rip)
	movb	$0, 50+carrinho(%rip)
	movb	$0, 51+carrinho(%rip)
	movb	$0, 52+carrinho(%rip)
	movb	$0, 53+carrinho(%rip)
	movb	$0, 54+carrinho(%rip)
	movb	$0, 55+carrinho(%rip)
	movb	$0, 56+carrinho(%rip)
	movb	$0, 57+carrinho(%rip)
	movb	$0, 58+carrinho(%rip)
	movb	$0, 59+carrinho(%rip)
	movb	$0, 60+carrinho(%rip)
	movb	$0, 61+carrinho(%rip)
	movb	$0, 62+carrinho(%rip)
	movb	$0, 63+carrinho(%rip)
	movb	$0, 64+carrinho(%rip)
	movb	$0, 65+carrinho(%rip)
	movb	$0, 66+carrinho(%rip)
	movb	$0, 67+carrinho(%rip)
	movb	$0, 68+carrinho(%rip)
	movb	$0, 69+carrinho(%rip)
	movb	$0, 70+carrinho(%rip)
	movb	$0, 71+carrinho(%rip)
	movb	$0, 72+carrinho(%rip)
	movb	$0, 73+carrinho(%rip)
	movb	$0, 74+carrinho(%rip)
	movb	$0, 75+carrinho(%rip)
	movb	$0, 76+carrinho(%rip)
	movb	$0, 77+carrinho(%rip)
	pxor	%xmm0, %xmm0
	movss	%xmm0, 80+carrinho(%rip)
	movl	$0, 84+carrinho(%rip)
	movl	$0, 88+carrinho(%rip)
	movb	$0, 92+carrinho(%rip)
	movb	$0, 93+carrinho(%rip)
	movb	$0, 94+carrinho(%rip)
	movb	$0, 95+carrinho(%rip)
	movb	$0, 96+carrinho(%rip)
	movb	$0, 97+carrinho(%rip)
	movb	$0, 98+carrinho(%rip)
	movb	$0, 99+carrinho(%rip)
	movb	$0, 100+carrinho(%rip)
	movb	$0, 101+carrinho(%rip)
	movb	$0, 102+carrinho(%rip)
	movb	$0, 103+carrinho(%rip)
	movb	$0, 104+carrinho(%rip)
	movb	$0, 105+carrinho(%rip)
	movb	$0, 106+carrinho(%rip)
	movb	$0, 107+carrinho(%rip)
	movb	$0, 108+carrinho(%rip)
	movb	$0, 109+carrinho(%rip)
	movb	$0, 110+carrinho(%rip)
	movb	$0, 111+carrinho(%rip)
	movb	$0, 112+carrinho(%rip)
	movb	$0, 113+carrinho(%rip)
	movb	$0, 114+carrinho(%rip)
	movb	$0, 115+carrinho(%rip)
	movb	$0, 116+carrinho(%rip)
	movb	$0, 117+carrinho(%rip)
	movb	$0, 118+carrinho(%rip)
	movb	$0, 119+carrinho(%rip)
	movb	$0, 120+carrinho(%rip)
	movb	$0, 121+carrinho(%rip)
	pxor	%xmm0, %xmm0
	movss	%xmm0, 124+carrinho(%rip)
	movl	$0, 128+carrinho(%rip)
	movl	$0, 132+carrinho(%rip)
	movb	$0, 136+carrinho(%rip)
	movb	$0, 137+carrinho(%rip)
	movb	$0, 138+carrinho(%rip)
	movb	$0, 139+carrinho(%rip)
	movb	$0, 140+carrinho(%rip)
	movb	$0, 141+carrinho(%rip)
	movb	$0, 142+carrinho(%rip)
	movb	$0, 143+carrinho(%rip)
	movb	$0, 144+carrinho(%rip)
	movb	$0, 145+carrinho(%rip)
	movb	$0, 146+carrinho(%rip)
	movb	$0, 147+carrinho(%rip)
	movb	$0, 148+carrinho(%rip)
	movb	$0, 149+carrinho(%rip)
	movb	$0, 150+carrinho(%rip)
	movb	$0, 151+carrinho(%rip)
	movb	$0, 152+carrinho(%rip)
	movb	$0, 153+carrinho(%rip)
	movb	$0, 154+carrinho(%rip)
	movb	$0, 155+carrinho(%rip)
	movb	$0, 156+carrinho(%rip)
	movb	$0, 157+carrinho(%rip)
	movb	$0, 158+carrinho(%rip)
	movb	$0, 159+carrinho(%rip)
	movb	$0, 160+carrinho(%rip)
	movb	$0, 161+carrinho(%rip)
	movb	$0, 162+carrinho(%rip)
	movb	$0, 163+carrinho(%rip)
	movb	$0, 164+carrinho(%rip)
	movb	$0, 165+carrinho(%rip)
	pxor	%xmm0, %xmm0
	movss	%xmm0, 168+carrinho(%rip)
	movl	$0, 172+carrinho(%rip)
	movl	$0, 176+carrinho(%rip)
	movb	$0, 180+carrinho(%rip)
	movb	$0, 181+carrinho(%rip)
	movb	$0, 182+carrinho(%rip)
	movb	$0, 183+carrinho(%rip)
	movb	$0, 184+carrinho(%rip)
	movb	$0, 185+carrinho(%rip)
	movb	$0, 186+carrinho(%rip)
	movb	$0, 187+carrinho(%rip)
	movb	$0, 188+carrinho(%rip)
	movb	$0, 189+carrinho(%rip)
	movb	$0, 190+carrinho(%rip)
	movb	$0, 191+carrinho(%rip)
	movb	$0, 192+carrinho(%rip)
	movb	$0, 193+carrinho(%rip)
	movb	$0, 194+carrinho(%rip)
	movb	$0, 195+carrinho(%rip)
	movb	$0, 196+carrinho(%rip)
	movb	$0, 197+carrinho(%rip)
	movb	$0, 198+carrinho(%rip)
	movb	$0, 199+carrinho(%rip)
	movb	$0, 200+carrinho(%rip)
	movb	$0, 201+carrinho(%rip)
	movb	$0, 202+carrinho(%rip)
	movb	$0, 203+carrinho(%rip)
	movb	$0, 204+carrinho(%rip)
	movb	$0, 205+carrinho(%rip)
	movb	$0, 206+carrinho(%rip)
	movb	$0, 207+carrinho(%rip)
	movb	$0, 208+carrinho(%rip)
	movb	$0, 209+carrinho(%rip)
	pxor	%xmm0, %xmm0
	movss	%xmm0, 212+carrinho(%rip)
	movl	$0, 216+carrinho(%rip)
	movl	$0, 220+carrinho(%rip)
	movb	$0, 224+carrinho(%rip)
	movb	$0, 225+carrinho(%rip)
	movb	$0, 226+carrinho(%rip)
	movb	$0, 227+carrinho(%rip)
	movb	$0, 228+carrinho(%rip)
	movb	$0, 229+carrinho(%rip)
	movb	$0, 230+carrinho(%rip)
	movb	$0, 231+carrinho(%rip)
	movb	$0, 232+carrinho(%rip)
	movb	$0, 233+carrinho(%rip)
	movb	$0, 234+carrinho(%rip)
	movb	$0, 235+carrinho(%rip)
	movb	$0, 236+carrinho(%rip)
	movb	$0, 237+carrinho(%rip)
	movb	$0, 238+carrinho(%rip)
	movb	$0, 239+carrinho(%rip)
	movb	$0, 240+carrinho(%rip)
	movb	$0, 241+carrinho(%rip)
	movb	$0, 242+carrinho(%rip)
	movb	$0, 243+carrinho(%rip)
	movb	$0, 244+carrinho(%rip)
	movb	$0, 245+carrinho(%rip)
	movb	$0, 246+carrinho(%rip)
	movb	$0, 247+carrinho(%rip)
	movb	$0, 248+carrinho(%rip)
	movb	$0, 249+carrinho(%rip)
	movb	$0, 250+carrinho(%rip)
	movb	$0, 251+carrinho(%rip)
	movb	$0, 252+carrinho(%rip)
	movb	$0, 253+carrinho(%rip)
	pxor	%xmm0, %xmm0
	movss	%xmm0, 256+carrinho(%rip)
	movl	$0, 260+carrinho(%rip)
	movl	$0, 264+carrinho(%rip)
	movb	$0, 268+carrinho(%rip)
	movb	$0, 269+carrinho(%rip)
	movb	$0, 270+carrinho(%rip)
	movb	$0, 271+carrinho(%rip)
	movb	$0, 272+carrinho(%rip)
	movb	$0, 273+carrinho(%rip)
	movb	$0, 274+carrinho(%rip)
	movb	$0, 275+carrinho(%rip)
	movb	$0, 276+carrinho(%rip)
	movb	$0, 277+carrinho(%rip)
	movb	$0, 278+carrinho(%rip)
	movb	$0, 279+carrinho(%rip)
	movb	$0, 280+carrinho(%rip)
	movb	$0, 281+carrinho(%rip)
	movb	$0, 282+carrinho(%rip)
	movb	$0, 283+carrinho(%rip)
	movb	$0, 284+carrinho(%rip)
	movb	$0, 285+carrinho(%rip)
	movb	$0, 286+carrinho(%rip)
	movb	$0, 287+carrinho(%rip)
	movb	$0, 288+carrinho(%rip)
	movb	$0, 289+carrinho(%rip)
	movb	$0, 290+carrinho(%rip)
	movb	$0, 291+carrinho(%rip)
	movb	$0, 292+carrinho(%rip)
	movb	$0, 293+carrinho(%rip)
	movb	$0, 294+carrinho(%rip)
	movb	$0, 295+carrinho(%rip)
	movb	$0, 296+carrinho(%rip)
	movb	$0, 297+carrinho(%rip)
	pxor	%xmm0, %xmm0
	movss	%xmm0, 300+carrinho(%rip)
	movl	$0, 304+carrinho(%rip)
	movl	$0, 308+carrinho(%rip)
	movb	$0, 312+carrinho(%rip)
	movb	$0, 313+carrinho(%rip)
	movb	$0, 314+carrinho(%rip)
	movb	$0, 315+carrinho(%rip)
	movb	$0, 316+carrinho(%rip)
	movb	$0, 317+carrinho(%rip)
	movb	$0, 318+carrinho(%rip)
	movb	$0, 319+carrinho(%rip)
	movb	$0, 320+carrinho(%rip)
	movb	$0, 321+carrinho(%rip)
	movb	$0, 322+carrinho(%rip)
	movb	$0, 323+carrinho(%rip)
	movb	$0, 324+carrinho(%rip)
	movb	$0, 325+carrinho(%rip)
	movb	$0, 326+carrinho(%rip)
	movb	$0, 327+carrinho(%rip)
	movb	$0, 328+carrinho(%rip)
	movb	$0, 329+carrinho(%rip)
	movb	$0, 330+carrinho(%rip)
	movb	$0, 331+carrinho(%rip)
	movb	$0, 332+carrinho(%rip)
	movb	$0, 333+carrinho(%rip)
	movb	$0, 334+carrinho(%rip)
	movb	$0, 335+carrinho(%rip)
	movb	$0, 336+carrinho(%rip)
	movb	$0, 337+carrinho(%rip)
	movb	$0, 338+carrinho(%rip)
	movb	$0, 339+carrinho(%rip)
	movb	$0, 340+carrinho(%rip)
	movb	$0, 341+carrinho(%rip)
	pxor	%xmm0, %xmm0
	movss	%xmm0, 344+carrinho(%rip)
	movl	$0, 348+carrinho(%rip)
	movl	$0, 352+carrinho(%rip)
	movb	$0, 356+carrinho(%rip)
	movb	$0, 357+carrinho(%rip)
	movb	$0, 358+carrinho(%rip)
	movb	$0, 359+carrinho(%rip)
	movb	$0, 360+carrinho(%rip)
	movb	$0, 361+carrinho(%rip)
	movb	$0, 362+carrinho(%rip)
	movb	$0, 363+carrinho(%rip)
	movb	$0, 364+carrinho(%rip)
	movb	$0, 365+carrinho(%rip)
	movb	$0, 366+carrinho(%rip)
	movb	$0, 367+carrinho(%rip)
	movb	$0, 368+carrinho(%rip)
	movb	$0, 369+carrinho(%rip)
	movb	$0, 370+carrinho(%rip)
	movb	$0, 371+carrinho(%rip)
	movb	$0, 372+carrinho(%rip)
	movb	$0, 373+carrinho(%rip)
	movb	$0, 374+carrinho(%rip)
	movb	$0, 375+carrinho(%rip)
	movb	$0, 376+carrinho(%rip)
	movb	$0, 377+carrinho(%rip)
	movb	$0, 378+carrinho(%rip)
	movb	$0, 379+carrinho(%rip)
	movb	$0, 380+carrinho(%rip)
	movb	$0, 381+carrinho(%rip)
	movb	$0, 382+carrinho(%rip)
	movb	$0, 383+carrinho(%rip)
	movb	$0, 384+carrinho(%rip)
	movb	$0, 385+carrinho(%rip)
	pxor	%xmm0, %xmm0
	movss	%xmm0, 388+carrinho(%rip)
	movl	$0, 392+carrinho(%rip)
	movl	$0, 396+carrinho(%rip)
	movb	$0, 400+carrinho(%rip)
	movb	$0, 401+carrinho(%rip)
	movb	$0, 402+carrinho(%rip)
	movb	$0, 403+carrinho(%rip)
	movb	$0, 404+carrinho(%rip)
	movb	$0, 405+carrinho(%rip)
	movb	$0, 406+carrinho(%rip)
	movb	$0, 407+carrinho(%rip)
	movb	$0, 408+carrinho(%rip)
	movb	$0, 409+carrinho(%rip)
	movb	$0, 410+carrinho(%rip)
	movb	$0, 411+carrinho(%rip)
	movb	$0, 412+carrinho(%rip)
	movb	$0, 413+carrinho(%rip)
	movb	$0, 414+carrinho(%rip)
	movb	$0, 415+carrinho(%rip)
	movb	$0, 416+carrinho(%rip)
	movb	$0, 417+carrinho(%rip)
	movb	$0, 418+carrinho(%rip)
	movb	$0, 419+carrinho(%rip)
	movb	$0, 420+carrinho(%rip)
	movb	$0, 421+carrinho(%rip)
	movb	$0, 422+carrinho(%rip)
	movb	$0, 423+carrinho(%rip)
	movb	$0, 424+carrinho(%rip)
	movb	$0, 425+carrinho(%rip)
	movb	$0, 426+carrinho(%rip)
	movb	$0, 427+carrinho(%rip)
	movb	$0, 428+carrinho(%rip)
	movb	$0, 429+carrinho(%rip)
	pxor	%xmm0, %xmm0
	movss	%xmm0, 432+carrinho(%rip)
	movl	$0, 436+carrinho(%rip)
	nop
.L4:
	movl	$0, estoque(%rip)
	movb	$0, 4+estoque(%rip)
	movb	$0, 5+estoque(%rip)
	movb	$0, 6+estoque(%rip)
	movb	$0, 7+estoque(%rip)
	movb	$0, 8+estoque(%rip)
	movb	$0, 9+estoque(%rip)
	movb	$0, 10+estoque(%rip)
	movb	$0, 11+estoque(%rip)
	movb	$0, 12+estoque(%rip)
	movb	$0, 13+estoque(%rip)
	movb	$0, 14+estoque(%rip)
	movb	$0, 15+estoque(%rip)
	movb	$0, 16+estoque(%rip)
	movb	$0, 17+estoque(%rip)
	movb	$0, 18+estoque(%rip)
	movb	$0, 19+estoque(%rip)
	movb	$0, 20+estoque(%rip)
	movb	$0, 21+estoque(%rip)
	movb	$0, 22+estoque(%rip)
	movb	$0, 23+estoque(%rip)
	movb	$0, 24+estoque(%rip)
	movb	$0, 25+estoque(%rip)
	movb	$0, 26+estoque(%rip)
	movb	$0, 27+estoque(%rip)
	movb	$0, 28+estoque(%rip)
	movb	$0, 29+estoque(%rip)
	movb	$0, 30+estoque(%rip)
	movb	$0, 31+estoque(%rip)
	movb	$0, 32+estoque(%rip)
	movb	$0, 33+estoque(%rip)
	pxor	%xmm0, %xmm0
	movss	%xmm0, 36+estoque(%rip)
	movl	$0, 40+estoque(%rip)
	movb	$0, 44+estoque(%rip)
	movb	$0, 45+estoque(%rip)
	movb	$0, 46+estoque(%rip)
	movb	$0, 47+estoque(%rip)
	movb	$0, 48+estoque(%rip)
	movb	$0, 49+estoque(%rip)
	movb	$0, 50+estoque(%rip)
	movb	$0, 51+estoque(%rip)
	movb	$0, 52+estoque(%rip)
	movb	$0, 53+estoque(%rip)
	movb	$0, 54+estoque(%rip)
	movb	$0, 55+estoque(%rip)
	movb	$0, 56+estoque(%rip)
	movb	$0, 57+estoque(%rip)
	movb	$0, 58+estoque(%rip)
	movb	$0, 59+estoque(%rip)
	movb	$0, 60+estoque(%rip)
	movb	$0, 61+estoque(%rip)
	movb	$0, 62+estoque(%rip)
	movb	$0, 63+estoque(%rip)
	movb	$0, 64+estoque(%rip)
	movb	$0, 65+estoque(%rip)
	movb	$0, 66+estoque(%rip)
	movb	$0, 67+estoque(%rip)
	movb	$0, 68+estoque(%rip)
	movb	$0, 69+estoque(%rip)
	movb	$0, 70+estoque(%rip)
	movb	$0, 71+estoque(%rip)
	movb	$0, 72+estoque(%rip)
	movb	$0, 73+estoque(%rip)
	pxor	%xmm0, %xmm0
	movss	%xmm0, 76+estoque(%rip)
	movl	$0, 80+estoque(%rip)
	movb	$0, 84+estoque(%rip)
	movb	$0, 85+estoque(%rip)
	movb	$0, 86+estoque(%rip)
	movb	$0, 87+estoque(%rip)
	movb	$0, 88+estoque(%rip)
	movb	$0, 89+estoque(%rip)
	movb	$0, 90+estoque(%rip)
	movb	$0, 91+estoque(%rip)
	movb	$0, 92+estoque(%rip)
	movb	$0, 93+estoque(%rip)
	movb	$0, 94+estoque(%rip)
	movb	$0, 95+estoque(%rip)
	movb	$0, 96+estoque(%rip)
	movb	$0, 97+estoque(%rip)
	movb	$0, 98+estoque(%rip)
	movb	$0, 99+estoque(%rip)
	movb	$0, 100+estoque(%rip)
	movb	$0, 101+estoque(%rip)
	movb	$0, 102+estoque(%rip)
	movb	$0, 103+estoque(%rip)
	movb	$0, 104+estoque(%rip)
	movb	$0, 105+estoque(%rip)
	movb	$0, 106+estoque(%rip)
	movb	$0, 107+estoque(%rip)
	movb	$0, 108+estoque(%rip)
	movb	$0, 109+estoque(%rip)
	movb	$0, 110+estoque(%rip)
	movb	$0, 111+estoque(%rip)
	movb	$0, 112+estoque(%rip)
	movb	$0, 113+estoque(%rip)
	pxor	%xmm0, %xmm0
	movss	%xmm0, 116+estoque(%rip)
	movl	$0, 120+estoque(%rip)
	movb	$0, 124+estoque(%rip)
	movb	$0, 125+estoque(%rip)
	movb	$0, 126+estoque(%rip)
	movb	$0, 127+estoque(%rip)
	movb	$0, 128+estoque(%rip)
	movb	$0, 129+estoque(%rip)
	movb	$0, 130+estoque(%rip)
	movb	$0, 131+estoque(%rip)
	movb	$0, 132+estoque(%rip)
	movb	$0, 133+estoque(%rip)
	movb	$0, 134+estoque(%rip)
	movb	$0, 135+estoque(%rip)
	movb	$0, 136+estoque(%rip)
	movb	$0, 137+estoque(%rip)
	movb	$0, 138+estoque(%rip)
	movb	$0, 139+estoque(%rip)
	movb	$0, 140+estoque(%rip)
	movb	$0, 141+estoque(%rip)
	movb	$0, 142+estoque(%rip)
	movb	$0, 143+estoque(%rip)
	movb	$0, 144+estoque(%rip)
	movb	$0, 145+estoque(%rip)
	movb	$0, 146+estoque(%rip)
	movb	$0, 147+estoque(%rip)
	movb	$0, 148+estoque(%rip)
	movb	$0, 149+estoque(%rip)
	movb	$0, 150+estoque(%rip)
	movb	$0, 151+estoque(%rip)
	movb	$0, 152+estoque(%rip)
	movb	$0, 153+estoque(%rip)
	pxor	%xmm0, %xmm0
	movss	%xmm0, 156+estoque(%rip)
	movl	$0, 160+estoque(%rip)
	movb	$0, 164+estoque(%rip)
	movb	$0, 165+estoque(%rip)
	movb	$0, 166+estoque(%rip)
	movb	$0, 167+estoque(%rip)
	movb	$0, 168+estoque(%rip)
	movb	$0, 169+estoque(%rip)
	movb	$0, 170+estoque(%rip)
	movb	$0, 171+estoque(%rip)
	movb	$0, 172+estoque(%rip)
	movb	$0, 173+estoque(%rip)
	movb	$0, 174+estoque(%rip)
	movb	$0, 175+estoque(%rip)
	movb	$0, 176+estoque(%rip)
	movb	$0, 177+estoque(%rip)
	movb	$0, 178+estoque(%rip)
	movb	$0, 179+estoque(%rip)
	movb	$0, 180+estoque(%rip)
	movb	$0, 181+estoque(%rip)
	movb	$0, 182+estoque(%rip)
	movb	$0, 183+estoque(%rip)
	movb	$0, 184+estoque(%rip)
	movb	$0, 185+estoque(%rip)
	movb	$0, 186+estoque(%rip)
	movb	$0, 187+estoque(%rip)
	movb	$0, 188+estoque(%rip)
	movb	$0, 189+estoque(%rip)
	movb	$0, 190+estoque(%rip)
	movb	$0, 191+estoque(%rip)
	movb	$0, 192+estoque(%rip)
	movb	$0, 193+estoque(%rip)
	pxor	%xmm0, %xmm0
	movss	%xmm0, 196+estoque(%rip)
	movl	$0, 200+estoque(%rip)
	movb	$0, 204+estoque(%rip)
	movb	$0, 205+estoque(%rip)
	movb	$0, 206+estoque(%rip)
	movb	$0, 207+estoque(%rip)
	movb	$0, 208+estoque(%rip)
	movb	$0, 209+estoque(%rip)
	movb	$0, 210+estoque(%rip)
	movb	$0, 211+estoque(%rip)
	movb	$0, 212+estoque(%rip)
	movb	$0, 213+estoque(%rip)
	movb	$0, 214+estoque(%rip)
	movb	$0, 215+estoque(%rip)
	movb	$0, 216+estoque(%rip)
	movb	$0, 217+estoque(%rip)
	movb	$0, 218+estoque(%rip)
	movb	$0, 219+estoque(%rip)
	movb	$0, 220+estoque(%rip)
	movb	$0, 221+estoque(%rip)
	movb	$0, 222+estoque(%rip)
	movb	$0, 223+estoque(%rip)
	movb	$0, 224+estoque(%rip)
	movb	$0, 225+estoque(%rip)
	movb	$0, 226+estoque(%rip)
	movb	$0, 227+estoque(%rip)
	movb	$0, 228+estoque(%rip)
	movb	$0, 229+estoque(%rip)
	movb	$0, 230+estoque(%rip)
	movb	$0, 231+estoque(%rip)
	movb	$0, 232+estoque(%rip)
	movb	$0, 233+estoque(%rip)
	pxor	%xmm0, %xmm0
	movss	%xmm0, 236+estoque(%rip)
	movl	$0, 240+estoque(%rip)
	movb	$0, 244+estoque(%rip)
	movb	$0, 245+estoque(%rip)
	movb	$0, 246+estoque(%rip)
	movb	$0, 247+estoque(%rip)
	movb	$0, 248+estoque(%rip)
	movb	$0, 249+estoque(%rip)
	movb	$0, 250+estoque(%rip)
	movb	$0, 251+estoque(%rip)
	movb	$0, 252+estoque(%rip)
	movb	$0, 253+estoque(%rip)
	movb	$0, 254+estoque(%rip)
	movb	$0, 255+estoque(%rip)
	movb	$0, 256+estoque(%rip)
	movb	$0, 257+estoque(%rip)
	movb	$0, 258+estoque(%rip)
	movb	$0, 259+estoque(%rip)
	movb	$0, 260+estoque(%rip)
	movb	$0, 261+estoque(%rip)
	movb	$0, 262+estoque(%rip)
	movb	$0, 263+estoque(%rip)
	movb	$0, 264+estoque(%rip)
	movb	$0, 265+estoque(%rip)
	movb	$0, 266+estoque(%rip)
	movb	$0, 267+estoque(%rip)
	movb	$0, 268+estoque(%rip)
	movb	$0, 269+estoque(%rip)
	movb	$0, 270+estoque(%rip)
	movb	$0, 271+estoque(%rip)
	movb	$0, 272+estoque(%rip)
	movb	$0, 273+estoque(%rip)
	pxor	%xmm0, %xmm0
	movss	%xmm0, 276+estoque(%rip)
	movl	$0, 280+estoque(%rip)
	movb	$0, 284+estoque(%rip)
	movb	$0, 285+estoque(%rip)
	movb	$0, 286+estoque(%rip)
	movb	$0, 287+estoque(%rip)
	movb	$0, 288+estoque(%rip)
	movb	$0, 289+estoque(%rip)
	movb	$0, 290+estoque(%rip)
	movb	$0, 291+estoque(%rip)
	movb	$0, 292+estoque(%rip)
	movb	$0, 293+estoque(%rip)
	movb	$0, 294+estoque(%rip)
	movb	$0, 295+estoque(%rip)
	movb	$0, 296+estoque(%rip)
	movb	$0, 297+estoque(%rip)
	movb	$0, 298+estoque(%rip)
	movb	$0, 299+estoque(%rip)
	movb	$0, 300+estoque(%rip)
	movb	$0, 301+estoque(%rip)
	movb	$0, 302+estoque(%rip)
	movb	$0, 303+estoque(%rip)
	movb	$0, 304+estoque(%rip)
	movb	$0, 305+estoque(%rip)
	movb	$0, 306+estoque(%rip)
	movb	$0, 307+estoque(%rip)
	movb	$0, 308+estoque(%rip)
	movb	$0, 309+estoque(%rip)
	movb	$0, 310+estoque(%rip)
	movb	$0, 311+estoque(%rip)
	movb	$0, 312+estoque(%rip)
	movb	$0, 313+estoque(%rip)
	pxor	%xmm0, %xmm0
	movss	%xmm0, 316+estoque(%rip)
	movl	$0, 320+estoque(%rip)
	movb	$0, 324+estoque(%rip)
	movb	$0, 325+estoque(%rip)
	movb	$0, 326+estoque(%rip)
	movb	$0, 327+estoque(%rip)
	movb	$0, 328+estoque(%rip)
	movb	$0, 329+estoque(%rip)
	movb	$0, 330+estoque(%rip)
	movb	$0, 331+estoque(%rip)
	movb	$0, 332+estoque(%rip)
	movb	$0, 333+estoque(%rip)
	movb	$0, 334+estoque(%rip)
	movb	$0, 335+estoque(%rip)
	movb	$0, 336+estoque(%rip)
	movb	$0, 337+estoque(%rip)
	movb	$0, 338+estoque(%rip)
	movb	$0, 339+estoque(%rip)
	movb	$0, 340+estoque(%rip)
	movb	$0, 341+estoque(%rip)
	movb	$0, 342+estoque(%rip)
	movb	$0, 343+estoque(%rip)
	movb	$0, 344+estoque(%rip)
	movb	$0, 345+estoque(%rip)
	movb	$0, 346+estoque(%rip)
	movb	$0, 347+estoque(%rip)
	movb	$0, 348+estoque(%rip)
	movb	$0, 349+estoque(%rip)
	movb	$0, 350+estoque(%rip)
	movb	$0, 351+estoque(%rip)
	movb	$0, 352+estoque(%rip)
	movb	$0, 353+estoque(%rip)
	pxor	%xmm0, %xmm0
	movss	%xmm0, 356+estoque(%rip)
	movl	$0, 360+estoque(%rip)
	movb	$0, 364+estoque(%rip)
	movb	$0, 365+estoque(%rip)
	movb	$0, 366+estoque(%rip)
	movb	$0, 367+estoque(%rip)
	movb	$0, 368+estoque(%rip)
	movb	$0, 369+estoque(%rip)
	movb	$0, 370+estoque(%rip)
	movb	$0, 371+estoque(%rip)
	movb	$0, 372+estoque(%rip)
	movb	$0, 373+estoque(%rip)
	movb	$0, 374+estoque(%rip)
	movb	$0, 375+estoque(%rip)
	movb	$0, 376+estoque(%rip)
	movb	$0, 377+estoque(%rip)
	movb	$0, 378+estoque(%rip)
	movb	$0, 379+estoque(%rip)
	movb	$0, 380+estoque(%rip)
	movb	$0, 381+estoque(%rip)
	movb	$0, 382+estoque(%rip)
	movb	$0, 383+estoque(%rip)
	movb	$0, 384+estoque(%rip)
	movb	$0, 385+estoque(%rip)
	movb	$0, 386+estoque(%rip)
	movb	$0, 387+estoque(%rip)
	movb	$0, 388+estoque(%rip)
	movb	$0, 389+estoque(%rip)
	movb	$0, 390+estoque(%rip)
	movb	$0, 391+estoque(%rip)
	movb	$0, 392+estoque(%rip)
	movb	$0, 393+estoque(%rip)
	pxor	%xmm0, %xmm0
	movss	%xmm0, 396+estoque(%rip)
	nop
.L5:
	movq	$0, _TIG_IZ_q5CY_envp(%rip)
	nop
.L6:
	movq	$0, _TIG_IZ_q5CY_argv(%rip)
	nop
.L7:
	movl	$0, _TIG_IZ_q5CY_argc(%rip)
	nop
	nop
.L8:
.L9:
#APP
# 781 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-q5CY--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_q5CY_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_q5CY_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_q5CY_envp(%rip)
	nop
	movq	$4, -16(%rbp)
.L35:
	cmpq	$17, -16(%rbp)
	ja	.L38
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L12(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L12(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L12:
	.long	.L22-.L12
	.long	.L38-.L12
	.long	.L21-.L12
	.long	.L20-.L12
	.long	.L19-.L12
	.long	.L18-.L12
	.long	.L17-.L12
	.long	.L38-.L12
	.long	.L16-.L12
	.long	.L38-.L12
	.long	.L38-.L12
	.long	.L38-.L12
	.long	.L15-.L12
	.long	.L38-.L12
	.long	.L38-.L12
	.long	.L14-.L12
	.long	.L13-.L12
	.long	.L11-.L12
	.text
.L19:
	movq	$17, -16(%rbp)
	jmp	.L23
.L14:
	call	visualizarCarrinho
	movq	$6, -16(%rbp)
	jmp	.L23
.L15:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$6, -16(%rbp)
	jmp	.L23
.L16:
	call	comprarProduto
	movq	$6, -16(%rbp)
	jmp	.L23
.L20:
	movl	-20(%rbp), %eax
	cmpl	$5, %eax
	ja	.L24
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L26(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L26(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L26:
	.long	.L24-.L26
	.long	.L30-.L26
	.long	.L29-.L26
	.long	.L28-.L26
	.long	.L27-.L26
	.long	.L25-.L26
	.text
.L25:
	movq	$12, -16(%rbp)
	jmp	.L31
.L27:
	movq	$15, -16(%rbp)
	jmp	.L31
.L28:
	movq	$8, -16(%rbp)
	jmp	.L31
.L29:
	movq	$0, -16(%rbp)
	jmp	.L31
.L30:
	movq	$16, -16(%rbp)
	jmp	.L31
.L24:
	movq	$2, -16(%rbp)
	nop
.L31:
	jmp	.L23
.L13:
	call	cadastrarProduto
	movq	$6, -16(%rbp)
	jmp	.L23
.L11:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-20(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$3, -16(%rbp)
	jmp	.L23
.L17:
	movl	-20(%rbp), %eax
	cmpl	$5, %eax
	je	.L32
	movq	$17, -16(%rbp)
	jmp	.L23
.L32:
	movq	$5, -16(%rbp)
	jmp	.L23
.L18:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L36
	jmp	.L37
.L22:
	call	listarProdutos
	movq	$6, -16(%rbp)
	jmp	.L23
.L21:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$6, -16(%rbp)
	jmp	.L23
.L38:
	nop
.L23:
	jmp	.L35
.L37:
	call	__stack_chk_fail@PLT
.L36:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	main, .-main
	.section	.rodata
.LC10:
	.string	"Produtos cadastrados:"
	.align 8
.LC11:
	.string	"Codigo: %d, Nome: %s, Preco: %.2f\n"
	.text
	.globl	listarProdutos
	.type	listarProdutos, @function
listarProdutos:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$1, -8(%rbp)
.L51:
	cmpq	$4, -8(%rbp)
	ja	.L52
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L42(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L42(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L42:
	.long	.L46-.L42
	.long	.L45-.L42
	.long	.L44-.L42
	.long	.L43-.L42
	.long	.L53-.L42
	.text
.L45:
	movq	$3, -8(%rbp)
	jmp	.L48
.L43:
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$0, -12(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L48
.L46:
	movl	total_produtos(%rip), %eax
	cmpl	%eax, -12(%rbp)
	jge	.L49
	movq	$2, -8(%rbp)
	jmp	.L48
.L49:
	movq	$4, -8(%rbp)
	jmp	.L48
.L44:
	movl	-12(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	leaq	36+estoque(%rip), %rax
	movss	(%rdx,%rax), %xmm0
	pxor	%xmm1, %xmm1
	cvtss2sd	%xmm0, %xmm1
	movq	%xmm1, %rdx
	movl	-12(%rbp), %eax
	movslq	%eax, %rcx
	movq	%rcx, %rax
	salq	$2, %rax
	addq	%rcx, %rax
	salq	$3, %rax
	leaq	estoque(%rip), %rcx
	addq	%rcx, %rax
	leaq	4(%rax), %rcx
	movl	-12(%rbp), %eax
	movslq	%eax, %rsi
	movq	%rsi, %rax
	salq	$2, %rax
	addq	%rsi, %rax
	salq	$3, %rax
	movq	%rax, %rsi
	leaq	estoque(%rip), %rax
	movl	(%rsi,%rax), %eax
	movq	%rdx, %xmm0
	movq	%rcx, %rdx
	movl	%eax, %esi
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	addl	$1, -12(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L48
.L52:
	nop
.L48:
	jmp	.L51
.L53:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	listarProdutos, .-listarProdutos
	.section	.rodata
.LC12:
	.string	"Produto nao encontrado!"
	.align 8
.LC13:
	.string	"Produto adicionado ao carrinho!"
	.align 8
.LC14:
	.string	"Digite o codigo do produto que deseja comprar: "
.LC15:
	.string	"Digite a quantidade: "
	.text
	.globl	comprarProduto
	.type	comprarProduto, @function
comprarProduto:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$6, -16(%rbp)
.L72:
	cmpq	$10, -16(%rbp)
	ja	.L75
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L57(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L57(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L57:
	.long	.L65-.L57
	.long	.L64-.L57
	.long	.L76-.L57
	.long	.L62-.L57
	.long	.L61-.L57
	.long	.L60-.L57
	.long	.L59-.L57
	.long	.L76-.L57
	.long	.L75-.L57
	.long	.L75-.L57
	.long	.L56-.L57
	.text
.L61:
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$7, -16(%rbp)
	jmp	.L66
.L64:
	movl	total_itens_carrinho(%rip), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rcx
	leaq	carrinho(%rip), %rdx
	movl	-20(%rbp), %eax
	movslq	%eax, %rsi
	movq	%rsi, %rax
	salq	$2, %rax
	addq	%rsi, %rax
	salq	$3, %rax
	movq	%rax, %rsi
	leaq	estoque(%rip), %rax
	movq	(%rsi,%rax), %r8
	movq	8(%rsi,%rax), %r9
	movq	%r8, (%rcx,%rdx)
	movq	%r9, 8(%rcx,%rdx)
	movq	16(%rsi,%rax), %r8
	movq	24(%rsi,%rax), %r9
	movq	%r8, 16(%rcx,%rdx)
	movq	%r9, 24(%rcx,%rdx)
	movq	32(%rsi,%rax), %rax
	movq	%rax, 32(%rcx,%rdx)
	movl	total_itens_carrinho(%rip), %eax
	movl	-24(%rbp), %ecx
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	leaq	40+carrinho(%rip), %rax
	movl	%ecx, (%rdx,%rax)
	movl	total_itens_carrinho(%rip), %eax
	addl	$1, %eax
	movl	%eax, total_itens_carrinho(%rip)
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$2, -16(%rbp)
	jmp	.L66
.L62:
	movl	total_produtos(%rip), %eax
	cmpl	%eax, -20(%rbp)
	jge	.L67
	movq	$5, -16(%rbp)
	jmp	.L66
.L67:
	movq	$4, -16(%rbp)
	jmp	.L66
.L59:
	movq	$10, -16(%rbp)
	jmp	.L66
.L60:
	movl	-20(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	leaq	estoque(%rip), %rax
	movl	(%rdx,%rax), %edx
	movl	-28(%rbp), %eax
	cmpl	%eax, %edx
	jne	.L69
	movq	$1, -16(%rbp)
	jmp	.L66
.L69:
	movq	$0, -16(%rbp)
	jmp	.L66
.L56:
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-28(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	$0, -20(%rbp)
	movq	$3, -16(%rbp)
	jmp	.L66
.L65:
	addl	$1, -20(%rbp)
	movq	$3, -16(%rbp)
	jmp	.L66
.L75:
	nop
.L66:
	jmp	.L72
.L76:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L74
	call	__stack_chk_fail@PLT
.L74:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	comprarProduto, .-comprarProduto
	.section	.rodata
.LC16:
	.string	"Produtos no carrinho:"
	.align 8
.LC17:
	.string	"Codigo: %d, Nome: %s, Quantidade: %d, Preco: %.2f\n"
	.text
	.globl	visualizarCarrinho
	.type	visualizarCarrinho, @function
visualizarCarrinho:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$6, -8(%rbp)
.L89:
	cmpq	$7, -8(%rbp)
	ja	.L90
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L80(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L80(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L80:
	.long	.L84-.L80
	.long	.L90-.L80
	.long	.L90-.L80
	.long	.L83-.L80
	.long	.L82-.L80
	.long	.L90-.L80
	.long	.L81-.L80
	.long	.L91-.L80
	.text
.L82:
	movl	total_itens_carrinho(%rip), %eax
	cmpl	%eax, -12(%rbp)
	jge	.L85
	movq	$0, -8(%rbp)
	jmp	.L87
.L85:
	movq	$7, -8(%rbp)
	jmp	.L87
.L83:
	leaq	.LC16(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$0, -12(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L87
.L81:
	movq	$3, -8(%rbp)
	jmp	.L87
.L84:
	movl	-12(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	leaq	36+carrinho(%rip), %rax
	movss	(%rdx,%rax), %xmm0
	pxor	%xmm1, %xmm1
	cvtss2sd	%xmm0, %xmm1
	movq	%xmm1, %rsi
	movl	-12(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	leaq	40+carrinho(%rip), %rax
	movl	(%rdx,%rax), %ecx
	movl	-12(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	leaq	carrinho(%rip), %rdx
	addq	%rdx, %rax
	leaq	4(%rax), %rdi
	movl	-12(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	leaq	carrinho(%rip), %rax
	movl	(%rdx,%rax), %eax
	movq	%rsi, %xmm0
	movq	%rdi, %rdx
	movl	%eax, %esi
	leaq	.LC17(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	addl	$1, -12(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L87
.L90:
	nop
.L87:
	jmp	.L89
.L91:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	visualizarCarrinho, .-visualizarCarrinho
	.section	.rodata
.LC18:
	.string	"Digite o codigo do produto: "
.LC19:
	.string	"Digite o nome do produto: "
.LC20:
	.string	"%s"
.LC21:
	.string	"Digite o preco do produto: "
.LC22:
	.string	"%f"
	.align 8
.LC23:
	.string	"Produto cadastrado com sucesso!"
.LC24:
	.string	"Estoque cheio!"
	.text
	.globl	cadastrarProduto
	.type	cadastrarProduto, @function
cadastrarProduto:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$72, %rsp
	.cfi_offset 3, -24
	movq	%fs:40, %rax
	movq	%rax, -24(%rbp)
	xorl	%eax, %eax
	movq	$0, -72(%rbp)
.L101:
	cmpq	$3, -72(%rbp)
	je	.L104
	cmpq	$3, -72(%rbp)
	ja	.L105
	cmpq	$2, -72(%rbp)
	je	.L95
	cmpq	$2, -72(%rbp)
	ja	.L105
	cmpq	$0, -72(%rbp)
	je	.L96
	cmpq	$1, -72(%rbp)
	jne	.L105
	leaq	.LC18(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-64(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC19(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-64(%rbp), %rax
	addq	$4, %rax
	movq	%rax, %rsi
	leaq	.LC20(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC21(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-64(%rbp), %rax
	addq	$36, %rax
	movq	%rax, %rsi
	leaq	.LC22(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	total_produtos(%rip), %eax
	movl	%eax, -76(%rbp)
	movl	total_produtos(%rip), %eax
	addl	$1, %eax
	movl	%eax, total_produtos(%rip)
	movl	-76(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	leaq	estoque(%rip), %rax
	movq	-64(%rbp), %rcx
	movq	-56(%rbp), %rbx
	movq	%rcx, (%rdx,%rax)
	movq	%rbx, 8(%rdx,%rax)
	movq	-48(%rbp), %rcx
	movq	-40(%rbp), %rbx
	movq	%rcx, 16(%rdx,%rax)
	movq	%rbx, 24(%rdx,%rax)
	movq	-32(%rbp), %rcx
	movq	%rcx, 32(%rdx,%rax)
	leaq	.LC23(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$3, -72(%rbp)
	jmp	.L97
.L96:
	movl	total_produtos(%rip), %eax
	cmpl	$9, %eax
	jg	.L99
	movq	$1, -72(%rbp)
	jmp	.L97
.L99:
	movq	$2, -72(%rbp)
	jmp	.L97
.L95:
	leaq	.LC24(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$3, -72(%rbp)
	jmp	.L97
.L105:
	nop
.L97:
	jmp	.L101
.L104:
	nop
	movq	-24(%rbp), %rax
	subq	%fs:40, %rax
	je	.L103
	call	__stack_chk_fail@PLT
.L103:
	movq	-8(%rbp), %rbx
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	cadastrarProduto, .-cadastrarProduto
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:

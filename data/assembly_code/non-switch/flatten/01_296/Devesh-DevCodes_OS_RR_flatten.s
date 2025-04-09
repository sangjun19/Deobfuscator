	.file	"Devesh-DevCodes_OS_RR_flatten.c"
	.text
	.globl	_TIG_IZ_SNsh_argv
	.bss
	.align 8
	.type	_TIG_IZ_SNsh_argv, @object
	.size	_TIG_IZ_SNsh_argv, 8
_TIG_IZ_SNsh_argv:
	.zero	8
	.globl	_TIG_IZ_SNsh_argc
	.align 4
	.type	_TIG_IZ_SNsh_argc, @object
	.size	_TIG_IZ_SNsh_argc, 4
_TIG_IZ_SNsh_argc:
	.zero	4
	.globl	quant
	.align 4
	.type	quant, @object
	.size	quant, 4
quant:
	.zero	4
	.globl	_TIG_IZ_SNsh_envp
	.align 8
	.type	_TIG_IZ_SNsh_envp, @object
	.size	_TIG_IZ_SNsh_envp, 8
_TIG_IZ_SNsh_envp:
	.zero	8
	.section	.rodata
	.align 8
.LC1:
	.string	"Enter the CPU burst times of processes: "
.LC2:
	.string	"Enter the no of processes: "
.LC3:
	.string	"%d"
.LC5:
	.string	"Enter Time Quantum: "
.LC8:
	.string	"Average WT = %lf\n"
.LC9:
	.string	"Average TAT = %lf\n"
	.align 8
.LC10:
	.string	"Enter the arrival times of processes: "
	.text
	.globl	main
	.type	main, @function
main:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$192, %rsp
	movl	%edi, -164(%rbp)
	movq	%rsi, -176(%rbp)
	movq	%rdx, -184(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movl	$0, quant(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_SNsh_envp(%rip)
	nop
.L3:
	movq	$0, _TIG_IZ_SNsh_argv(%rip)
	nop
.L4:
	movl	$0, _TIG_IZ_SNsh_argc(%rip)
	nop
	nop
.L5:
.L6:
#APP
# 155 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-SNsh--0
# 0 "" 2
#NO_APP
	movl	-164(%rbp), %eax
	movl	%eax, _TIG_IZ_SNsh_argc(%rip)
	movq	-176(%rbp), %rax
	movq	%rax, _TIG_IZ_SNsh_argv(%rip)
	movq	-184(%rbp), %rax
	movq	%rax, _TIG_IZ_SNsh_envp(%rip)
	nop
	movq	$16, -64(%rbp)
.L110:
	cmpq	$83, -64(%rbp)
	ja	.L124
	movq	-64(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L9(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L9(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L9:
	.long	.L57-.L9
	.long	.L124-.L9
	.long	.L56-.L9
	.long	.L124-.L9
	.long	.L55-.L9
	.long	.L124-.L9
	.long	.L124-.L9
	.long	.L54-.L9
	.long	.L53-.L9
	.long	.L124-.L9
	.long	.L124-.L9
	.long	.L52-.L9
	.long	.L124-.L9
	.long	.L124-.L9
	.long	.L124-.L9
	.long	.L51-.L9
	.long	.L50-.L9
	.long	.L124-.L9
	.long	.L124-.L9
	.long	.L49-.L9
	.long	.L48-.L9
	.long	.L124-.L9
	.long	.L47-.L9
	.long	.L46-.L9
	.long	.L45-.L9
	.long	.L124-.L9
	.long	.L44-.L9
	.long	.L43-.L9
	.long	.L42-.L9
	.long	.L124-.L9
	.long	.L124-.L9
	.long	.L41-.L9
	.long	.L40-.L9
	.long	.L39-.L9
	.long	.L124-.L9
	.long	.L38-.L9
	.long	.L37-.L9
	.long	.L124-.L9
	.long	.L36-.L9
	.long	.L124-.L9
	.long	.L124-.L9
	.long	.L124-.L9
	.long	.L35-.L9
	.long	.L34-.L9
	.long	.L33-.L9
	.long	.L32-.L9
	.long	.L31-.L9
	.long	.L124-.L9
	.long	.L30-.L9
	.long	.L124-.L9
	.long	.L29-.L9
	.long	.L124-.L9
	.long	.L124-.L9
	.long	.L124-.L9
	.long	.L28-.L9
	.long	.L27-.L9
	.long	.L26-.L9
	.long	.L124-.L9
	.long	.L124-.L9
	.long	.L25-.L9
	.long	.L24-.L9
	.long	.L23-.L9
	.long	.L22-.L9
	.long	.L21-.L9
	.long	.L20-.L9
	.long	.L124-.L9
	.long	.L124-.L9
	.long	.L124-.L9
	.long	.L19-.L9
	.long	.L124-.L9
	.long	.L18-.L9
	.long	.L17-.L9
	.long	.L16-.L9
	.long	.L15-.L9
	.long	.L14-.L9
	.long	.L124-.L9
	.long	.L13-.L9
	.long	.L12-.L9
	.long	.L124-.L9
	.long	.L11-.L9
	.long	.L124-.L9
	.long	.L10-.L9
	.long	.L124-.L9
	.long	.L8-.L9
	.text
.L29:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L111
	jmp	.L118
.L55:
	movl	-120(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-88(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm0
	pxor	%xmm1, %xmm1
	ucomiss	%xmm1, %xmm0
	jp	.L59
	pxor	%xmm1, %xmm1
	ucomiss	%xmm1, %xmm0
	jne	.L59
	movq	$63, -64(%rbp)
	jmp	.L62
.L59:
	movq	$71, -64(%rbp)
	jmp	.L62
.L22:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$0, -140(%rbp)
	movq	$79, -64(%rbp)
	jmp	.L62
.L51:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-144(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$83, -64(%rbp)
	jmp	.L62
.L26:
	movss	-128(%rbp), %xmm1
	movss	.LC4(%rip), %xmm0
	addss	%xmm1, %xmm0
	movss	%xmm0, -128(%rbp)
	movq	$71, -64(%rbp)
	jmp	.L62
.L11:
	movl	-144(%rbp), %eax
	cmpl	%eax, -140(%rbp)
	jge	.L63
	movq	$38, -64(%rbp)
	jmp	.L62
.L63:
	movq	$81, -64(%rbp)
	jmp	.L62
.L41:
	movl	-136(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movl	-140(%rbp), %eax
	cltq
	imulq	-72(%rbp), %rax
	addq	%rax, %rdx
	movq	-96(%rbp), %rax
	addq	%rdx, %rax
	movl	$-1, (%rax)
	addl	$1, -136(%rbp)
	movq	$46, -64(%rbp)
	jmp	.L62
.L53:
	movl	-120(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-88(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm1
	movl	quant(%rip), %eax
	pxor	%xmm0, %xmm0
	cvtsi2ssl	%eax, %xmm0
	comiss	%xmm1, %xmm0
	jb	.L119
	movq	$33, -64(%rbp)
	jmp	.L62
.L119:
	movq	$59, -64(%rbp)
	jmp	.L62
.L32:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	quant(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	$0, -140(%rbp)
	movq	$42, -64(%rbp)
	jmp	.L62
.L28:
	movss	.LC6(%rip), %xmm0
	movss	%xmm0, -124(%rbp)
	movb	$0, -145(%rbp)
	movl	$0, -140(%rbp)
	movq	$28, -64(%rbp)
	jmp	.L62
.L10:
	movl	-144(%rbp), %eax
	movl	%eax, -132(%rbp)
	movq	$24, -64(%rbp)
	jmp	.L62
.L46:
	addl	$1, -136(%rbp)
	movq	$36, -64(%rbp)
	jmp	.L62
.L12:
	pxor	%xmm0, %xmm0
	movss	%xmm0, -128(%rbp)
	movss	.LC6(%rip), %xmm0
	movss	%xmm0, -124(%rbp)
	movq	$60, -64(%rbp)
	jmp	.L62
.L18:
	movl	-140(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-80(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm1
	movss	-124(%rbp), %xmm0
	comiss	%xmm1, %xmm0
	jbe	.L120
	movq	$48, -64(%rbp)
	jmp	.L62
.L120:
	movq	$72, -64(%rbp)
	jmp	.L62
.L50:
	movq	$15, -64(%rbp)
	jmp	.L62
.L45:
	movl	-144(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$7, %rax
	addq	$31, %rax
	shrq	$3, %rax
	movq	%rax, %rdx
	movabsq	$2305843009213693948, %rax
	andq	%rdx, %rax
	movq	%rax, -40(%rbp)
	movq	$80, -72(%rbp)
	movq	-40(%rbp), %rax
	leaq	8(%rax), %rdx
	movl	$16, %eax
	subq	$1, %rax
	addq	%rdx, %rax
	movl	$16, %esi
	movl	$0, %edx
	divq	%rsi
	imulq	$16, %rax, %rax
	movq	%rax, %rcx
	andq	$-4096, %rcx
	movq	%rsp, %rdx
	subq	%rcx, %rdx
.L71:
	cmpq	%rdx, %rsp
	je	.L72
	subq	$4096, %rsp
	orq	$0, 4088(%rsp)
	jmp	.L71
.L72:
	movq	%rax, %rdx
	andl	$4095, %edx
	subq	%rdx, %rsp
	movq	%rax, %rdx
	andl	$4095, %edx
	testq	%rdx, %rdx
	je	.L73
	andl	$4095, %eax
	subq	$8, %rax
	addq	%rsp, %rax
	orq	$0, (%rax)
.L73:
	movq	%rsp, %rax
	addq	$15, %rax
	shrq	$4, %rax
	salq	$4, %rax
	movq	%rax, -96(%rbp)
	movq	$77, -64(%rbp)
	jmp	.L62
.L37:
	movl	-136(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movl	-120(%rbp), %eax
	cltq
	imulq	-72(%rbp), %rax
	addq	%rax, %rdx
	movq	-96(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	$-1, %eax
	je	.L74
	movq	$23, -64(%rbp)
	jmp	.L62
.L74:
	movq	$55, -64(%rbp)
	jmp	.L62
.L13:
	pxor	%xmm1, %xmm1
	cvtss2sd	-128(%rbp), %xmm1
	movsd	.LC7(%rip), %xmm0
	addsd	%xmm1, %xmm0
	movl	-120(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-80(%rbp), %rax
	addq	%rdx, %rax
	cvtsd2ss	%xmm0, %xmm0
	movss	%xmm0, (%rax)
	movq	$4, -64(%rbp)
	jmp	.L62
.L19:
	movl	-120(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-88(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm0
	pxor	%xmm1, %xmm1
	comiss	%xmm1, %xmm0
	jbe	.L121
	movq	$76, -64(%rbp)
	jmp	.L62
.L121:
	movq	$4, -64(%rbp)
	jmp	.L62
.L44:
	movl	-140(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-104(%rbp), %rax
	addq	%rdx, %rax
	movl	-140(%rbp), %edx
	addl	$1, %edx
	movl	%edx, 100(%rax)
	addl	$1, -140(%rbp)
	movq	$42, -64(%rbp)
	jmp	.L62
.L52:
	movl	-144(%rbp), %eax
	cmpl	%eax, -140(%rbp)
	jge	.L79
	movq	$32, -64(%rbp)
	jmp	.L62
.L79:
	movq	$62, -64(%rbp)
	jmp	.L62
.L21:
	subl	$1, -132(%rbp)
	movl	-120(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-104(%rbp), %rax
	addq	%rax, %rdx
	movss	-128(%rbp), %xmm0
	cvttss2sil	%xmm0, %eax
	movl	%eax, 92(%rdx)
	movl	-120(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-104(%rbp), %rax
	addq	%rdx, %rax
	movl	92(%rax), %ecx
	movl	-120(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-104(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	subl	%eax, %ecx
	movl	-120(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-104(%rbp), %rax
	addq	%rdx, %rax
	movl	4(%rax), %esi
	movl	-120(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-104(%rbp), %rax
	addq	%rdx, %rax
	subl	%esi, %ecx
	movl	%ecx, %edx
	movl	%edx, 88(%rax)
	movl	-120(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-104(%rbp), %rax
	addq	%rdx, %rax
	movl	88(%rax), %eax
	addl	%eax, -116(%rbp)
	movl	-120(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-104(%rbp), %rax
	addq	%rdx, %rax
	movl	4(%rax), %esi
	movl	-120(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-104(%rbp), %rax
	addq	%rdx, %rax
	movl	88(%rax), %ecx
	movl	-120(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-104(%rbp), %rax
	addq	%rdx, %rax
	leal	(%rsi,%rcx), %edx
	movl	%edx, 96(%rax)
	movl	-120(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-104(%rbp), %rax
	addq	%rdx, %rax
	movl	96(%rax), %eax
	addl	%eax, -112(%rbp)
	movq	$71, -64(%rbp)
	jmp	.L62
.L49:
	pxor	%xmm0, %xmm0
	cvtsi2ssl	-116(%rbp), %xmm0
	movl	-144(%rbp), %eax
	pxor	%xmm1, %xmm1
	cvtsi2ssl	%eax, %xmm1
	divss	%xmm1, %xmm0
	cvtss2sd	%xmm0, %xmm0
	movsd	%xmm0, -24(%rbp)
	pxor	%xmm0, %xmm0
	cvtsi2ssl	-112(%rbp), %xmm0
	movl	-144(%rbp), %eax
	pxor	%xmm1, %xmm1
	cvtsi2ssl	%eax, %xmm1
	divss	%xmm1, %xmm0
	cvtss2sd	%xmm0, %xmm0
	movsd	%xmm0, -16(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, %xmm0
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	-16(%rbp), %rax
	movq	%rax, %xmm0
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$50, -64(%rbp)
	jmp	.L62
.L40:
	movl	-140(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-104(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	addl	$1, -140(%rbp)
	movq	$11, -64(%rbp)
	jmp	.L62
.L27:
	movl	-136(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movl	-120(%rbp), %eax
	cltq
	imulq	-72(%rbp), %rax
	addq	%rax, %rdx
	movq	-96(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	$-1, %eax
	jne	.L81
	movq	$74, -64(%rbp)
	jmp	.L62
.L81:
	movq	$8, -64(%rbp)
	jmp	.L62
.L24:
	movl	-144(%rbp), %eax
	cltq
	salq	$5, %rax
	addq	$31, %rax
	shrq	$3, %rax
	movq	%rax, %rdx
	movabsq	$2305843009213693948, %rax
	andq	%rdx, %rax
	movq	%rax, -48(%rbp)
	movq	-48(%rbp), %rax
	leaq	8(%rax), %rdx
	movl	$16, %eax
	subq	$1, %rax
	addq	%rdx, %rax
	movl	$16, %esi
	movl	$0, %edx
	divq	%rsi
	imulq	$16, %rax, %rax
	movq	%rax, %rcx
	andq	$-4096, %rcx
	movq	%rsp, %rdx
	subq	%rcx, %rdx
.L83:
	cmpq	%rdx, %rsp
	je	.L84
	subq	$4096, %rsp
	orq	$0, 4088(%rsp)
	jmp	.L83
.L84:
	movq	%rax, %rdx
	andl	$4095, %edx
	subq	%rdx, %rsp
	movq	%rax, %rdx
	andl	$4095, %edx
	testq	%rdx, %rdx
	je	.L85
	andl	$4095, %eax
	subq	$8, %rax
	addq	%rsp, %rax
	orq	$0, (%rax)
.L85:
	movq	%rsp, %rax
	addq	$15, %rax
	shrq	$4, %rax
	salq	$4, %rax
	movq	%rax, -88(%rbp)
	movq	$20, -64(%rbp)
	jmp	.L62
.L25:
	movl	quant(%rip), %eax
	pxor	%xmm0, %xmm0
	cvtsi2ssl	%eax, %xmm0
	movss	-128(%rbp), %xmm1
	addss	%xmm1, %xmm0
	movss	%xmm0, -128(%rbp)
	movl	-120(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-88(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm0
	movl	quant(%rip), %eax
	pxor	%xmm1, %xmm1
	cvtsi2ssl	%eax, %xmm1
	movl	-120(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-88(%rbp), %rax
	addq	%rdx, %rax
	subss	%xmm1, %xmm0
	movss	%xmm0, (%rax)
	movq	$68, -64(%rbp)
	jmp	.L62
.L43:
	movl	$-1, -120(%rbp)
	movl	$0, -140(%rbp)
	movq	$0, -64(%rbp)
	jmp	.L62
.L36:
	movl	-140(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-104(%rbp), %rax
	addq	%rdx, %rax
	addq	$4, %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	addl	$1, -140(%rbp)
	movq	$79, -64(%rbp)
	jmp	.L62
.L23:
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$0, -140(%rbp)
	movq	$11, -64(%rbp)
	jmp	.L62
.L14:
	movl	-136(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movl	-120(%rbp), %eax
	cltq
	imulq	-72(%rbp), %rax
	addq	%rax, %rdx
	movq	-96(%rbp), %rax
	addq	%rax, %rdx
	movss	-128(%rbp), %xmm0
	cvttss2sil	%xmm0, %eax
	movl	%eax, (%rdx)
	movl	-120(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-104(%rbp), %rax
	leaq	(%rdx,%rax), %rcx
	movss	-128(%rbp), %xmm0
	cvttss2sil	%xmm0, %edx
	movl	-136(%rbp), %eax
	cltq
	movl	%edx, 8(%rcx,%rax,4)
	movq	$8, -64(%rbp)
	jmp	.L62
.L30:
	movl	-140(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-88(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm0
	pxor	%xmm1, %xmm1
	comiss	%xmm1, %xmm0
	jbe	.L122
	movq	$43, -64(%rbp)
	jmp	.L62
.L122:
	movq	$72, -64(%rbp)
	jmp	.L62
.L17:
	cmpl	$0, -132(%rbp)
	je	.L89
	movq	$54, -64(%rbp)
	jmp	.L62
.L89:
	movq	$19, -64(%rbp)
	jmp	.L62
.L47:
	movl	-140(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-104(%rbp), %rax
	addq	%rdx, %rax
	movl	4(%rax), %edx
	movl	-140(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rcx
	movq	-88(%rbp), %rax
	addq	%rcx, %rax
	pxor	%xmm0, %xmm0
	cvtsi2ssl	%edx, %xmm0
	movss	%xmm0, (%rax)
	movl	-140(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-104(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	movl	-140(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rcx
	movq	-80(%rbp), %rax
	addq	%rcx, %rax
	pxor	%xmm0, %xmm0
	cvtsi2ssl	%edx, %xmm0
	movss	%xmm0, (%rax)
	movl	$0, -136(%rbp)
	movq	$46, -64(%rbp)
	jmp	.L62
.L42:
	movl	-144(%rbp), %eax
	cmpl	%eax, -140(%rbp)
	jge	.L91
	movq	$35, -64(%rbp)
	jmp	.L62
.L91:
	movq	$7, -64(%rbp)
	jmp	.L62
.L15:
	addl	$1, -140(%rbp)
	movq	$0, -64(%rbp)
	jmp	.L62
.L33:
	movl	$0, -136(%rbp)
	movq	$36, -64(%rbp)
	jmp	.L62
.L16:
	addl	$1, -140(%rbp)
	movq	$28, -64(%rbp)
	jmp	.L62
.L39:
	movl	-120(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-88(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm0
	movss	-128(%rbp), %xmm1
	addss	%xmm1, %xmm0
	movss	%xmm0, -128(%rbp)
	movl	-120(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-88(%rbp), %rax
	addq	%rdx, %rax
	pxor	%xmm0, %xmm0
	movss	%xmm0, (%rax)
	movq	$68, -64(%rbp)
	jmp	.L62
.L20:
	movl	-140(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-80(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm1
	movss	-108(%rbp), %xmm0
	comiss	%xmm1, %xmm0
	jb	.L123
	movq	$70, -64(%rbp)
	jmp	.L62
.L123:
	movq	$72, -64(%rbp)
	jmp	.L62
.L35:
	movl	-144(%rbp), %eax
	cmpl	%eax, -140(%rbp)
	jge	.L96
	movq	$26, -64(%rbp)
	jmp	.L62
.L96:
	movq	$61, -64(%rbp)
	jmp	.L62
.L57:
	movl	-144(%rbp), %eax
	cmpl	%eax, -140(%rbp)
	jge	.L98
	movq	$22, -64(%rbp)
	jmp	.L62
.L98:
	movq	$2, -64(%rbp)
	jmp	.L62
.L31:
	cmpl	$19, -136(%rbp)
	jg	.L100
	movq	$31, -64(%rbp)
	jmp	.L62
.L100:
	movq	$73, -64(%rbp)
	jmp	.L62
.L8:
	movl	-144(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$6, %rax
	addq	$31, %rax
	shrq	$3, %rax
	movq	%rax, %rdx
	movabsq	$2305843009213693948, %rax
	andq	%rdx, %rax
	movq	%rax, -56(%rbp)
	movq	-56(%rbp), %rax
	leaq	8(%rax), %rdx
	movl	$16, %eax
	subq	$1, %rax
	addq	%rdx, %rax
	movl	$16, %edi
	movl	$0, %edx
	divq	%rdi
	imulq	$16, %rax, %rax
	movq	%rax, %rcx
	andq	$-4096, %rcx
	movq	%rsp, %rdx
	subq	%rcx, %rdx
.L102:
	cmpq	%rdx, %rsp
	je	.L103
	subq	$4096, %rsp
	orq	$0, 4088(%rsp)
	jmp	.L102
.L103:
	movq	%rax, %rdx
	andl	$4095, %edx
	subq	%rdx, %rsp
	movq	%rax, %rdx
	andl	$4095, %edx
	testq	%rdx, %rdx
	je	.L104
	andl	$4095, %eax
	subq	$8, %rax
	addq	%rsp, %rax
	orq	$0, (%rax)
.L104:
	movq	%rsp, %rax
	addq	$15, %rax
	shrq	$4, %rax
	salq	$4, %rax
	movq	%rax, -104(%rbp)
	movq	$45, -64(%rbp)
	jmp	.L62
.L54:
	movzbl	-145(%rbp), %eax
	xorl	$1, %eax
	testb	%al, %al
	je	.L105
	movq	$56, -64(%rbp)
	jmp	.L62
.L105:
	movq	$44, -64(%rbp)
	jmp	.L62
.L38:
	pxor	%xmm1, %xmm1
	cvtss2sd	-128(%rbp), %xmm1
	movsd	.LC7(%rip), %xmm0
	addsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm0
	movss	%xmm0, -108(%rbp)
	movq	$64, -64(%rbp)
	jmp	.L62
.L34:
	movl	-140(%rbp), %eax
	movl	%eax, -120(%rbp)
	movl	-140(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-80(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm0
	movss	%xmm0, -124(%rbp)
	movb	$1, -145(%rbp)
	movq	$72, -64(%rbp)
	jmp	.L62
.L56:
	movl	$0, -116(%rbp)
	movl	$0, -112(%rbp)
	movb	$0, -145(%rbp)
	movq	$71, -64(%rbp)
	jmp	.L62
.L48:
	movl	-144(%rbp), %eax
	cltq
	salq	$5, %rax
	addq	$31, %rax
	shrq	$3, %rax
	movq	%rax, %rdx
	movabsq	$2305843009213693948, %rax
	andq	%rdx, %rax
	movq	%rax, -32(%rbp)
	movq	-32(%rbp), %rax
	leaq	8(%rax), %rdx
	movl	$16, %eax
	subq	$1, %rax
	addq	%rdx, %rax
	movl	$16, %esi
	movl	$0, %edx
	divq	%rsi
	imulq	$16, %rax, %rax
	movq	%rax, %rcx
	andq	$-4096, %rcx
	movq	%rsp, %rdx
	subq	%rcx, %rdx
.L107:
	cmpq	%rdx, %rsp
	je	.L108
	subq	$4096, %rsp
	orq	$0, 4088(%rsp)
	jmp	.L107
.L108:
	movq	%rax, %rdx
	andl	$4095, %edx
	subq	%rdx, %rsp
	movq	%rax, %rdx
	andl	$4095, %edx
	testq	%rdx, %rdx
	je	.L109
	andl	$4095, %eax
	subq	$8, %rax
	addq	%rsp, %rax
	orq	$0, (%rax)
.L109:
	movq	%rsp, %rax
	addq	$15, %rax
	shrq	$4, %rax
	salq	$4, %rax
	movq	%rax, -80(%rbp)
	movq	$27, -64(%rbp)
	jmp	.L62
.L124:
	nop
.L62:
	jmp	.L110
.L118:
	call	__stack_chk_fail@PLT
.L111:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	main, .-main
	.section	.rodata
	.align 4
.LC4:
	.long	1065353216
	.align 4
.LC6:
	.long	1325400064
	.align 8
.LC7:
	.long	-1717986918
	.long	1069128089
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

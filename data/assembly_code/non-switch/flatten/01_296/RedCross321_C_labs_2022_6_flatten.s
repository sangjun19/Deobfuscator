	.file	"RedCross321_C_labs_2022_6_flatten.c"
	.text
	.globl	_TIG_IZ_wSHC_argv
	.bss
	.align 8
	.type	_TIG_IZ_wSHC_argv, @object
	.size	_TIG_IZ_wSHC_argv, 8
_TIG_IZ_wSHC_argv:
	.zero	8
	.globl	_TIG_IZ_wSHC_envp
	.align 8
	.type	_TIG_IZ_wSHC_envp, @object
	.size	_TIG_IZ_wSHC_envp, 8
_TIG_IZ_wSHC_envp:
	.zero	8
	.globl	_TIG_IZ_wSHC_argc
	.align 4
	.type	_TIG_IZ_wSHC_argc, @object
	.size	_TIG_IZ_wSHC_argc, 4
_TIG_IZ_wSHC_argc:
	.zero	4
	.text
	.globl	fill
	.type	fill, @function
fill:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -20(%rbp)
	movq	%rsi, -32(%rbp)
	movq	$6, -8(%rbp)
.L11:
	cmpq	$6, -8(%rbp)
	je	.L2
	cmpq	$6, -8(%rbp)
	ja	.L13
	cmpq	$4, -8(%rbp)
	je	.L4
	cmpq	$4, -8(%rbp)
	ja	.L13
	cmpq	$0, -8(%rbp)
	je	.L5
	cmpq	$2, -8(%rbp)
	je	.L6
	jmp	.L13
.L4:
	movl	-16(%rbp), %eax
	cmpl	-20(%rbp), %eax
	jge	.L7
	movq	$0, -8(%rbp)
	jmp	.L9
.L7:
	movq	$2, -8(%rbp)
	jmp	.L9
.L2:
	movl	$0, -16(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L9
.L5:
	call	rand@PLT
	movl	%eax, -12(%rbp)
	movl	-12(%rbp), %eax
	movslq	%eax, %rdx
	imulq	$680390859, %rdx, %rdx
	shrq	$32, %rdx
	sarl	$4, %edx
	movl	%eax, %ecx
	sarl	$31, %ecx
	subl	%ecx, %edx
	imull	$101, %edx, %ecx
	subl	%ecx, %eax
	movl	%eax, %edx
	movl	-16(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rcx
	movq	-32(%rbp), %rax
	addq	%rcx, %rax
	subl	$50, %edx
	movl	%edx, (%rax)
	addl	$1, -16(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L9
.L6:
	movl	$0, %eax
	jmp	.L12
.L13:
	nop
.L9:
	jmp	.L11
.L12:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	fill, .-fill
	.section	.rodata
.LC0:
	.string	"%d  "
.LC1:
	.string	"n -> "
.LC2:
	.string	"%d"
	.align 8
.LC3:
	.string	"\320\236\321\201\320\275\320\276\320\262\320\275\320\276\320\271 \320\274\320\260\321\201\321\201\320\270\320\262: "
	.text
	.globl	main
	.type	main, @function
main:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	subq	$192, %rsp
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	movl	%edi, -164(%rbp)
	movq	%rsi, -176(%rbp)
	movq	%rdx, -184(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -40(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_wSHC_envp(%rip)
	nop
.L15:
	movq	$0, _TIG_IZ_wSHC_argv(%rip)
	nop
.L16:
	movl	$0, _TIG_IZ_wSHC_argc(%rip)
	nop
	nop
.L17:
.L18:
#APP
# 135 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-wSHC--0
# 0 "" 2
#NO_APP
	movl	-164(%rbp), %eax
	movl	%eax, _TIG_IZ_wSHC_argc(%rip)
	movq	-176(%rbp), %rax
	movq	%rax, _TIG_IZ_wSHC_argv(%rip)
	movq	-184(%rbp), %rax
	movq	%rax, _TIG_IZ_wSHC_envp(%rip)
	nop
	movq	$0, -112(%rbp)
.L57:
	cmpq	$26, -112(%rbp)
	ja	.L60
	movq	-112(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L21(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L21(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L21:
	.long	.L37-.L21
	.long	.L36-.L21
	.long	.L35-.L21
	.long	.L34-.L21
	.long	.L60-.L21
	.long	.L60-.L21
	.long	.L33-.L21
	.long	.L60-.L21
	.long	.L60-.L21
	.long	.L60-.L21
	.long	.L60-.L21
	.long	.L32-.L21
	.long	.L31-.L21
	.long	.L30-.L21
	.long	.L29-.L21
	.long	.L28-.L21
	.long	.L27-.L21
	.long	.L26-.L21
	.long	.L25-.L21
	.long	.L60-.L21
	.long	.L60-.L21
	.long	.L24-.L21
	.long	.L23-.L21
	.long	.L22-.L21
	.long	.L60-.L21
	.long	.L60-.L21
	.long	.L20-.L21
	.text
.L25:
	movl	$0, %eax
	movq	-40(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L58
	jmp	.L59
.L29:
	movl	-156(%rbp), %eax
	movslq	%eax, %rdx
	subq	$1, %rdx
	movq	%rdx, -64(%rbp)
	cltq
	movq	%rax, %r12
	movl	$0, %r13d
	movl	-156(%rbp), %eax
	movq	-136(%rbp), %rdx
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	fill
	movl	$0, -144(%rbp)
	movq	$1, -112(%rbp)
	jmp	.L39
.L28:
	movl	-144(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-136(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	testl	%eax, %eax
	jne	.L40
	movq	$17, -112(%rbp)
	jmp	.L39
.L40:
	movq	$21, -112(%rbp)
	jmp	.L39
.L31:
	movl	-152(%rbp), %eax
	cltq
	salq	$5, %rax
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
	movl	$16, %esi
	movl	$0, %edx
	divq	%rsi
	imulq	$16, %rax, %rax
	movq	%rax, %rcx
	andq	$-4096, %rcx
	movq	%rsp, %rdx
	subq	%rcx, %rdx
.L42:
	cmpq	%rdx, %rsp
	je	.L43
	subq	$4096, %rsp
	orq	$0, 4088(%rsp)
	jmp	.L42
.L43:
	movq	%rax, %rdx
	andl	$4095, %edx
	subq	%rdx, %rsp
	movq	%rax, %rdx
	andl	$4095, %edx
	testq	%rdx, %rdx
	je	.L44
	andl	$4095, %eax
	subq	$8, %rax
	addq	%rsp, %rax
	orq	$0, (%rax)
.L44:
	movq	%rsp, %rax
	addq	$15, %rax
	shrq	$4, %rax
	salq	$4, %rax
	movq	%rax, -128(%rbp)
	movq	$16, -112(%rbp)
	jmp	.L39
.L36:
	movl	-156(%rbp), %eax
	cmpl	%eax, -144(%rbp)
	jge	.L45
	movq	$11, -112(%rbp)
	jmp	.L39
.L45:
	movq	$12, -112(%rbp)
	jmp	.L39
.L22:
	addl	$1, -152(%rbp)
	movq	$21, -112(%rbp)
	jmp	.L39
.L34:
	movl	-156(%rbp), %eax
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
.L47:
	cmpq	%rdx, %rsp
	je	.L48
	subq	$4096, %rsp
	orq	$0, 4088(%rsp)
	jmp	.L47
.L48:
	movq	%rax, %rdx
	andl	$4095, %edx
	subq	%rdx, %rsp
	movq	%rax, %rdx
	andl	$4095, %edx
	testq	%rdx, %rdx
	je	.L49
	andl	$4095, %eax
	subq	$8, %rax
	addq	%rsp, %rax
	orq	$0, (%rax)
.L49:
	movq	%rsp, %rax
	addq	$15, %rax
	shrq	$4, %rax
	salq	$4, %rax
	movq	%rax, -136(%rbp)
	movq	$14, -112(%rbp)
	jmp	.L39
.L27:
	movl	-156(%rbp), %eax
	subl	-152(%rbp), %eax
	subl	-148(%rbp), %eax
	cltq
	salq	$5, %rax
	addq	$31, %rax
	shrq	$3, %rax
	movq	%rax, %rdx
	movabsq	$2305843009213693948, %rax
	andq	%rdx, %rax
	movq	%rax, -72(%rbp)
	movq	-72(%rbp), %rax
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
.L50:
	cmpq	%rdx, %rsp
	je	.L51
	subq	$4096, %rsp
	orq	$0, 4088(%rsp)
	jmp	.L50
.L51:
	movq	%rax, %rdx
	andl	$4095, %edx
	subq	%rdx, %rsp
	movq	%rax, %rdx
	andl	$4095, %edx
	testq	%rdx, %rdx
	je	.L52
	andl	$4095, %eax
	subq	$8, %rax
	addq	%rsp, %rax
	orq	$0, (%rax)
.L52:
	movq	%rsp, %rax
	addq	$15, %rax
	shrq	$4, %rax
	salq	$4, %rax
	movq	%rax, -120(%rbp)
	movq	$2, -112(%rbp)
	jmp	.L39
.L24:
	addl	$1, -144(%rbp)
	movq	$1, -112(%rbp)
	jmp	.L39
.L20:
	movl	-156(%rbp), %eax
	subl	-152(%rbp), %eax
	subl	-148(%rbp), %eax
	movslq	%eax, %rdx
	subq	$1, %rdx
	movq	%rdx, -104(%rbp)
	cltq
	movq	%rax, %r14
	movl	$0, %r15d
	movl	-152(%rbp), %eax
	movslq	%eax, %rdx
	subq	$1, %rdx
	movq	%rdx, -96(%rbp)
	cltq
	movq	%rax, -208(%rbp)
	movq	$0, -200(%rbp)
	movl	-156(%rbp), %eax
	movslq	%eax, %rdx
	subq	$1, %rdx
	movq	%rdx, -88(%rbp)
	cltq
	movq	%rax, -224(%rbp)
	movq	$0, -216(%rbp)
	movl	-156(%rbp), %eax
	movq	-120(%rbp), %rcx
	movq	-128(%rbp), %rdx
	movq	-136(%rbp), %rsi
	movl	%eax, %edi
	call	raskid
	movl	$10, %edi
	call	putchar@PLT
	movq	$18, -112(%rbp)
	jmp	.L39
.L32:
	movl	-144(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-136(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	testl	%eax, %eax
	jle	.L53
	movq	$23, -112(%rbp)
	jmp	.L39
.L53:
	movq	$15, -112(%rbp)
	jmp	.L39
.L30:
	movl	-156(%rbp), %eax
	cmpl	%eax, -140(%rbp)
	jge	.L55
	movq	$6, -112(%rbp)
	jmp	.L39
.L55:
	movq	$26, -112(%rbp)
	jmp	.L39
.L26:
	addl	$1, -148(%rbp)
	movq	$21, -112(%rbp)
	jmp	.L39
.L33:
	movl	-140(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-136(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -140(%rbp)
	movq	$13, -112(%rbp)
	jmp	.L39
.L23:
	movl	$0, %edi
	call	time@PLT
	movq	%rax, -80(%rbp)
	movq	-80(%rbp), %rax
	movl	%eax, %edi
	call	srand@PLT
	movl	$0, -152(%rbp)
	movl	$0, -148(%rbp)
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-156(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$3, -112(%rbp)
	jmp	.L39
.L37:
	movq	$22, -112(%rbp)
	jmp	.L39
.L35:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$0, -140(%rbp)
	movq	$13, -112(%rbp)
	jmp	.L39
.L60:
	nop
.L39:
	jmp	.L57
.L59:
	call	__stack_chk_fail@PLT
.L58:
	leaq	-32(%rbp), %rsp
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	main, .-main
	.section	.rodata
	.align 8
.LC4:
	.string	"\n\320\237\320\276\320\273\320\276\320\266\320\270\321\202\320\265\320\273\321\214\320\275\321\213\320\271 \320\274\320\260\321\201\321\201\320\270\320\262: "
	.align 8
.LC5:
	.string	"\n\320\236\321\202\321\200\320\270\321\206\320\260\321\202\320\265\320\273\321\214\320\275\321\213\320\271 \320\274\320\260\321\201\321\201\320\270\320\262: "
	.text
	.globl	raskid
	.type	raskid, @function
raskid:
.LFB7:
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
	movq	%rcx, -64(%rbp)
	movq	$11, -8(%rbp)
.L91:
	cmpq	$23, -8(%rbp)
	ja	.L93
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L64(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L64(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L64:
	.long	.L78-.L64
	.long	.L77-.L64
	.long	.L76-.L64
	.long	.L93-.L64
	.long	.L93-.L64
	.long	.L75-.L64
	.long	.L74-.L64
	.long	.L73-.L64
	.long	.L72-.L64
	.long	.L93-.L64
	.long	.L71-.L64
	.long	.L70-.L64
	.long	.L69-.L64
	.long	.L68-.L64
	.long	.L67-.L64
	.long	.L66-.L64
	.long	.L93-.L64
	.long	.L93-.L64
	.long	.L93-.L64
	.long	.L93-.L64
	.long	.L93-.L64
	.long	.L65-.L64
	.long	.L93-.L64
	.long	.L63-.L64
	.text
.L67:
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-48(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	testl	%eax, %eax
	jns	.L79
	movq	$5, -8(%rbp)
	jmp	.L81
.L79:
	movq	$23, -8(%rbp)
	jmp	.L81
.L66:
	movl	$0, -28(%rbp)
	movl	$0, -24(%rbp)
	movl	$0, -20(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L81
.L69:
	movl	-16(%rbp), %eax
	cmpl	-28(%rbp), %eax
	jge	.L82
	movq	$10, -8(%rbp)
	jmp	.L81
.L82:
	movq	$1, -8(%rbp)
	jmp	.L81
.L72:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$0, -16(%rbp)
	movq	$12, -8(%rbp)
	jmp	.L81
.L77:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$0, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L81
.L63:
	addl	$1, -20(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L81
.L65:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-64(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L81
.L70:
	movq	$15, -8(%rbp)
	jmp	.L81
.L68:
	movl	$0, %eax
	jmp	.L92
.L74:
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-48(%rbp), %rax
	addq	%rdx, %rax
	movl	-28(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-56(%rbp), %rdx
	addq	%rcx, %rdx
	movl	(%rax), %eax
	movl	%eax, (%rdx)
	addl	$1, -28(%rbp)
	movq	$23, -8(%rbp)
	jmp	.L81
.L75:
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-48(%rbp), %rax
	addq	%rdx, %rax
	movl	-24(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-64(%rbp), %rdx
	addq	%rcx, %rdx
	movl	(%rax), %eax
	movl	%eax, (%rdx)
	addl	$1, -24(%rbp)
	movq	$23, -8(%rbp)
	jmp	.L81
.L71:
	movl	-16(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -16(%rbp)
	movq	$12, -8(%rbp)
	jmp	.L81
.L78:
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-48(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	testl	%eax, %eax
	jle	.L85
	movq	$6, -8(%rbp)
	jmp	.L81
.L85:
	movq	$14, -8(%rbp)
	jmp	.L81
.L73:
	movl	-20(%rbp), %eax
	cmpl	-36(%rbp), %eax
	jge	.L87
	movq	$0, -8(%rbp)
	jmp	.L81
.L87:
	movq	$8, -8(%rbp)
	jmp	.L81
.L76:
	movl	-12(%rbp), %eax
	cmpl	-24(%rbp), %eax
	jge	.L89
	movq	$21, -8(%rbp)
	jmp	.L81
.L89:
	movq	$13, -8(%rbp)
	jmp	.L81
.L93:
	nop
.L81:
	jmp	.L91
.L92:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	raskid, .-raskid
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

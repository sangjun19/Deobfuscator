	.file	"lac-dcc_jotai-benchmarks_extr_linuxdriversmediaplatformexynos-gscgsc-core.c_gsc_set_frame_size_Final_flatten.c"
	.text
	.globl	_TIG_IZ_rKHp_envp
	.bss
	.align 8
	.type	_TIG_IZ_rKHp_envp, @object
	.size	_TIG_IZ_rKHp_envp, 8
_TIG_IZ_rKHp_envp:
	.zero	8
	.globl	rand_primes
	.align 32
	.type	rand_primes, @object
	.size	rand_primes, 100
rand_primes:
	.zero	100
	.globl	_TIG_IZ_rKHp_argc
	.align 4
	.type	_TIG_IZ_rKHp_argc, @object
	.size	_TIG_IZ_rKHp_argc, 4
_TIG_IZ_rKHp_argc:
	.zero	4
	.globl	_TIG_IZ_rKHp_argv
	.align 8
	.type	_TIG_IZ_rKHp_argv, @object
	.size	_TIG_IZ_rKHp_argv, 8
_TIG_IZ_rKHp_argv:
	.zero	8
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
	subq	$384, %rsp
	movl	%edi, -356(%rbp)
	movq	%rsi, -368(%rbp)
	movq	%rdx, -376(%rbp)
	movl	$179, rand_primes(%rip)
	movl	$103, 4+rand_primes(%rip)
	movl	$479, 8+rand_primes(%rip)
	movl	$647, 12+rand_primes(%rip)
	movl	$229, 16+rand_primes(%rip)
	movl	$37, 20+rand_primes(%rip)
	movl	$271, 24+rand_primes(%rip)
	movl	$557, 28+rand_primes(%rip)
	movl	$263, 32+rand_primes(%rip)
	movl	$607, 36+rand_primes(%rip)
	movl	$18743, 40+rand_primes(%rip)
	movl	$50359, 44+rand_primes(%rip)
	movl	$21929, 48+rand_primes(%rip)
	movl	$48757, 52+rand_primes(%rip)
	movl	$98179, 56+rand_primes(%rip)
	movl	$12907, 60+rand_primes(%rip)
	movl	$52937, 64+rand_primes(%rip)
	movl	$64579, 68+rand_primes(%rip)
	movl	$49957, 72+rand_primes(%rip)
	movl	$52567, 76+rand_primes(%rip)
	movl	$507163, 80+rand_primes(%rip)
	movl	$149939, 84+rand_primes(%rip)
	movl	$412157, 88+rand_primes(%rip)
	movl	$680861, 92+rand_primes(%rip)
	movl	$757751, 96+rand_primes(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_rKHp_envp(%rip)
	nop
.L3:
	movq	$0, _TIG_IZ_rKHp_argv(%rip)
	nop
.L4:
	movl	$0, _TIG_IZ_rKHp_argc(%rip)
	nop
	nop
.L5:
.L6:
#APP
# 230 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-rKHp--0
# 0 "" 2
#NO_APP
	movl	-356(%rbp), %eax
	movl	%eax, _TIG_IZ_rKHp_argc(%rip)
	movq	-368(%rbp), %rax
	movq	%rax, _TIG_IZ_rKHp_argv(%rip)
	movq	-376(%rbp), %rax
	movq	%rax, _TIG_IZ_rKHp_envp(%rip)
	nop
	movq	$2, -40(%rbp)
.L50:
	cmpq	$44, -40(%rbp)
	ja	.L51
	movq	-40(%rbp), %rax
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
	.long	.L51-.L9
	.long	.L51-.L9
	.long	.L31-.L9
	.long	.L30-.L9
	.long	.L29-.L9
	.long	.L51-.L9
	.long	.L28-.L9
	.long	.L51-.L9
	.long	.L51-.L9
	.long	.L27-.L9
	.long	.L51-.L9
	.long	.L26-.L9
	.long	.L25-.L9
	.long	.L51-.L9
	.long	.L51-.L9
	.long	.L24-.L9
	.long	.L51-.L9
	.long	.L23-.L9
	.long	.L22-.L9
	.long	.L21-.L9
	.long	.L51-.L9
	.long	.L51-.L9
	.long	.L51-.L9
	.long	.L51-.L9
	.long	.L51-.L9
	.long	.L20-.L9
	.long	.L51-.L9
	.long	.L19-.L9
	.long	.L18-.L9
	.long	.L51-.L9
	.long	.L51-.L9
	.long	.L17-.L9
	.long	.L16-.L9
	.long	.L15-.L9
	.long	.L14-.L9
	.long	.L13-.L9
	.long	.L51-.L9
	.long	.L51-.L9
	.long	.L51-.L9
	.long	.L51-.L9
	.long	.L12-.L9
	.long	.L11-.L9
	.long	.L51-.L9
	.long	.L10-.L9
	.long	.L8-.L9
	.text
.L22:
	movl	-328(%rbp), %edx
	movl	-332(%rbp), %ecx
	movq	-64(%rbp), %rax
	movl	%ecx, %esi
	movq	%rax, %rdi
	call	gsc_set_frame_size
	movq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$19, -40(%rbp)
	jmp	.L32
.L20:
	call	next_i
	movl	%eax, -264(%rbp)
	call	next_i
	movl	%eax, -260(%rbp)
	movl	-264(%rbp), %eax
	cltd
	shrl	$31, %edx
	addl	%edx, %eax
	andl	$1, %eax
	subl	%edx, %eax
	movl	%eax, %edx
	movl	$0, %eax
	subl	%edx, %eax
	addl	%eax, %eax
	addl	$1, %eax
	movl	-320(%rbp), %edx
	movslq	%edx, %rdx
	movq	%rdx, %rcx
	salq	$5, %rcx
	movq	-64(%rbp), %rdx
	addq	%rcx, %rdx
	imull	-260(%rbp), %eax
	movl	%eax, (%rdx)
	call	next_i
	movl	%eax, -256(%rbp)
	call	next_i
	movl	%eax, -252(%rbp)
	movl	-256(%rbp), %eax
	cltd
	shrl	$31, %edx
	addl	%edx, %eax
	andl	$1, %eax
	subl	%edx, %eax
	movl	%eax, %edx
	movl	$0, %eax
	subl	%edx, %eax
	addl	%eax, %eax
	addl	$1, %eax
	movl	-320(%rbp), %edx
	movslq	%edx, %rdx
	movq	%rdx, %rcx
	salq	$5, %rcx
	movq	-64(%rbp), %rdx
	addq	%rcx, %rdx
	imull	-252(%rbp), %eax
	movl	%eax, 4(%rdx)
	call	next_i
	movl	%eax, -248(%rbp)
	call	next_i
	movl	%eax, -244(%rbp)
	movl	-248(%rbp), %eax
	cltd
	shrl	$31, %edx
	addl	%edx, %eax
	andl	$1, %eax
	subl	%edx, %eax
	movl	%eax, %edx
	movl	$0, %eax
	subl	%edx, %eax
	addl	%eax, %eax
	addl	$1, %eax
	movl	-320(%rbp), %edx
	movslq	%edx, %rdx
	movq	%rdx, %rcx
	salq	$5, %rcx
	movq	-64(%rbp), %rdx
	addq	%rcx, %rdx
	imull	-244(%rbp), %eax
	movl	%eax, 8(%rdx)
	call	next_i
	movl	%eax, -240(%rbp)
	call	next_i
	movl	%eax, -236(%rbp)
	movl	-240(%rbp), %eax
	cltd
	shrl	$31, %edx
	addl	%edx, %eax
	andl	$1, %eax
	subl	%edx, %eax
	movl	%eax, %edx
	movl	$0, %eax
	subl	%edx, %eax
	addl	%eax, %eax
	addl	$1, %eax
	movl	-320(%rbp), %edx
	movslq	%edx, %rdx
	movq	%rdx, %rcx
	salq	$5, %rcx
	movq	-64(%rbp), %rdx
	addq	%rcx, %rdx
	imull	-236(%rbp), %eax
	movl	%eax, 12(%rdx)
	call	next_i
	movl	%eax, -232(%rbp)
	call	next_i
	movl	%eax, -228(%rbp)
	movl	-232(%rbp), %eax
	cltd
	shrl	$31, %edx
	addl	%edx, %eax
	andl	$1, %eax
	subl	%edx, %eax
	movl	%eax, %edx
	movl	$0, %eax
	subl	%edx, %eax
	addl	%eax, %eax
	addl	$1, %eax
	imull	-228(%rbp), %eax
	movl	-320(%rbp), %edx
	movslq	%edx, %rdx
	movq	%rdx, %rcx
	salq	$5, %rcx
	movq	-64(%rbp), %rdx
	addq	%rcx, %rdx
	cltq
	movq	%rax, 16(%rdx)
	call	next_i
	movl	%eax, -224(%rbp)
	call	next_i
	movl	%eax, -220(%rbp)
	movl	-224(%rbp), %eax
	cltd
	shrl	$31, %edx
	addl	%edx, %eax
	andl	$1, %eax
	subl	%edx, %eax
	movl	%eax, %edx
	movl	$0, %eax
	subl	%edx, %eax
	addl	%eax, %eax
	addl	$1, %eax
	imull	-220(%rbp), %eax
	movl	-320(%rbp), %edx
	movslq	%edx, %rdx
	movq	%rdx, %rcx
	salq	$5, %rcx
	movq	-64(%rbp), %rdx
	addq	%rcx, %rdx
	cltq
	movq	%rax, 24(%rdx)
	addl	$1, -320(%rbp)
	movq	$32, -40(%rbp)
	jmp	.L32
.L29:
	call	next_i
	movl	%eax, -120(%rbp)
	call	next_i
	movl	%eax, -116(%rbp)
	movl	-120(%rbp), %eax
	cltd
	shrl	$31, %edx
	addl	%edx, %eax
	andl	$1, %eax
	subl	%edx, %eax
	movl	%eax, %edx
	movl	$0, %eax
	subl	%edx, %eax
	addl	%eax, %eax
	addl	$1, %eax
	movl	-304(%rbp), %edx
	movslq	%edx, %rdx
	movq	%rdx, %rcx
	salq	$5, %rcx
	movq	-56(%rbp), %rdx
	addq	%rcx, %rdx
	imull	-116(%rbp), %eax
	movl	%eax, (%rdx)
	call	next_i
	movl	%eax, -112(%rbp)
	call	next_i
	movl	%eax, -108(%rbp)
	movl	-112(%rbp), %eax
	cltd
	shrl	$31, %edx
	addl	%edx, %eax
	andl	$1, %eax
	subl	%edx, %eax
	movl	%eax, %edx
	movl	$0, %eax
	subl	%edx, %eax
	addl	%eax, %eax
	addl	$1, %eax
	movl	-304(%rbp), %edx
	movslq	%edx, %rdx
	movq	%rdx, %rcx
	salq	$5, %rcx
	movq	-56(%rbp), %rdx
	addq	%rcx, %rdx
	imull	-108(%rbp), %eax
	movl	%eax, 4(%rdx)
	call	next_i
	movl	%eax, -104(%rbp)
	call	next_i
	movl	%eax, -100(%rbp)
	movl	-104(%rbp), %eax
	cltd
	shrl	$31, %edx
	addl	%edx, %eax
	andl	$1, %eax
	subl	%edx, %eax
	movl	%eax, %edx
	movl	$0, %eax
	subl	%edx, %eax
	addl	%eax, %eax
	addl	$1, %eax
	movl	-304(%rbp), %edx
	movslq	%edx, %rdx
	movq	%rdx, %rcx
	salq	$5, %rcx
	movq	-56(%rbp), %rdx
	addq	%rcx, %rdx
	imull	-100(%rbp), %eax
	movl	%eax, 8(%rdx)
	call	next_i
	movl	%eax, -96(%rbp)
	call	next_i
	movl	%eax, -92(%rbp)
	movl	-96(%rbp), %eax
	cltd
	shrl	$31, %edx
	addl	%edx, %eax
	andl	$1, %eax
	subl	%edx, %eax
	movl	%eax, %edx
	movl	$0, %eax
	subl	%edx, %eax
	addl	%eax, %eax
	addl	$1, %eax
	movl	-304(%rbp), %edx
	movslq	%edx, %rdx
	movq	%rdx, %rcx
	salq	$5, %rcx
	movq	-56(%rbp), %rdx
	addq	%rcx, %rdx
	imull	-92(%rbp), %eax
	movl	%eax, 12(%rdx)
	call	next_i
	movl	%eax, -88(%rbp)
	call	next_i
	movl	%eax, -84(%rbp)
	movl	-88(%rbp), %eax
	cltd
	shrl	$31, %edx
	addl	%edx, %eax
	andl	$1, %eax
	subl	%edx, %eax
	movl	%eax, %edx
	movl	$0, %eax
	subl	%edx, %eax
	addl	%eax, %eax
	addl	$1, %eax
	imull	-84(%rbp), %eax
	movl	-304(%rbp), %edx
	movslq	%edx, %rdx
	movq	%rdx, %rcx
	salq	$5, %rcx
	movq	-56(%rbp), %rdx
	addq	%rcx, %rdx
	cltq
	movq	%rax, 16(%rdx)
	call	next_i
	movl	%eax, -80(%rbp)
	call	next_i
	movl	%eax, -76(%rbp)
	movl	-80(%rbp), %eax
	cltd
	shrl	$31, %edx
	addl	%edx, %eax
	andl	$1, %eax
	subl	%edx, %eax
	movl	%eax, %edx
	movl	$0, %eax
	subl	%edx, %eax
	addl	%eax, %eax
	addl	$1, %eax
	imull	-76(%rbp), %eax
	movl	-304(%rbp), %edx
	movslq	%edx, %rdx
	movq	%rdx, %rcx
	salq	$5, %rcx
	movq	-56(%rbp), %rdx
	addq	%rcx, %rdx
	cltq
	movq	%rax, 24(%rdx)
	addl	$1, -304(%rbp)
	movq	$41, -40(%rbp)
	jmp	.L32
.L24:
	call	usage
	movq	$40, -40(%rbp)
	jmp	.L32
.L17:
	movq	-368(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movl	%eax, -284(%rbp)
	movl	-284(%rbp), %eax
	movl	%eax, -352(%rbp)
	movq	$35, -40(%rbp)
	jmp	.L32
.L25:
	movl	-344(%rbp), %edx
	movl	-348(%rbp), %ecx
	movq	-72(%rbp), %rax
	movl	%ecx, %esi
	movq	%rax, %rdi
	call	gsc_set_frame_size
	movq	-72(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$19, -40(%rbp)
	jmp	.L32
.L30:
	movl	$255, -332(%rbp)
	movl	$255, -328(%rbp)
	movl	$65025, -324(%rbp)
	movl	-324(%rbp), %eax
	cltq
	salq	$5, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -64(%rbp)
	movl	$0, -320(%rbp)
	movq	$32, -40(%rbp)
	jmp	.L32
.L26:
	call	usage
	movq	$19, -40(%rbp)
	jmp	.L32
.L27:
	call	next_i
	movl	%eax, -168(%rbp)
	call	next_i
	movl	%eax, -164(%rbp)
	movl	-168(%rbp), %eax
	cltd
	shrl	$31, %edx
	addl	%edx, %eax
	andl	$1, %eax
	subl	%edx, %eax
	movl	%eax, %edx
	movl	$0, %eax
	subl	%edx, %eax
	addl	%eax, %eax
	addl	$1, %eax
	movl	-336(%rbp), %edx
	movslq	%edx, %rdx
	movq	%rdx, %rcx
	salq	$5, %rcx
	movq	-72(%rbp), %rdx
	addq	%rcx, %rdx
	imull	-164(%rbp), %eax
	movl	%eax, (%rdx)
	call	next_i
	movl	%eax, -160(%rbp)
	call	next_i
	movl	%eax, -156(%rbp)
	movl	-160(%rbp), %eax
	cltd
	shrl	$31, %edx
	addl	%edx, %eax
	andl	$1, %eax
	subl	%edx, %eax
	movl	%eax, %edx
	movl	$0, %eax
	subl	%edx, %eax
	addl	%eax, %eax
	addl	$1, %eax
	movl	-336(%rbp), %edx
	movslq	%edx, %rdx
	movq	%rdx, %rcx
	salq	$5, %rcx
	movq	-72(%rbp), %rdx
	addq	%rcx, %rdx
	imull	-156(%rbp), %eax
	movl	%eax, 4(%rdx)
	call	next_i
	movl	%eax, -152(%rbp)
	call	next_i
	movl	%eax, -148(%rbp)
	movl	-152(%rbp), %eax
	cltd
	shrl	$31, %edx
	addl	%edx, %eax
	andl	$1, %eax
	subl	%edx, %eax
	movl	%eax, %edx
	movl	$0, %eax
	subl	%edx, %eax
	addl	%eax, %eax
	addl	$1, %eax
	movl	-336(%rbp), %edx
	movslq	%edx, %rdx
	movq	%rdx, %rcx
	salq	$5, %rcx
	movq	-72(%rbp), %rdx
	addq	%rcx, %rdx
	imull	-148(%rbp), %eax
	movl	%eax, 8(%rdx)
	call	next_i
	movl	%eax, -144(%rbp)
	call	next_i
	movl	%eax, -140(%rbp)
	movl	-144(%rbp), %eax
	cltd
	shrl	$31, %edx
	addl	%edx, %eax
	andl	$1, %eax
	subl	%edx, %eax
	movl	%eax, %edx
	movl	$0, %eax
	subl	%edx, %eax
	addl	%eax, %eax
	addl	$1, %eax
	movl	-336(%rbp), %edx
	movslq	%edx, %rdx
	movq	%rdx, %rcx
	salq	$5, %rcx
	movq	-72(%rbp), %rdx
	addq	%rcx, %rdx
	imull	-140(%rbp), %eax
	movl	%eax, 12(%rdx)
	call	next_i
	movl	%eax, -136(%rbp)
	call	next_i
	movl	%eax, -132(%rbp)
	movl	-136(%rbp), %eax
	cltd
	shrl	$31, %edx
	addl	%edx, %eax
	andl	$1, %eax
	subl	%edx, %eax
	movl	%eax, %edx
	movl	$0, %eax
	subl	%edx, %eax
	addl	%eax, %eax
	addl	$1, %eax
	imull	-132(%rbp), %eax
	movl	-336(%rbp), %edx
	movslq	%edx, %rdx
	movq	%rdx, %rcx
	salq	$5, %rcx
	movq	-72(%rbp), %rdx
	addq	%rcx, %rdx
	cltq
	movq	%rax, 16(%rdx)
	call	next_i
	movl	%eax, -128(%rbp)
	call	next_i
	movl	%eax, -124(%rbp)
	movl	-128(%rbp), %eax
	cltd
	shrl	$31, %edx
	addl	%edx, %eax
	andl	$1, %eax
	subl	%edx, %eax
	movl	%eax, %edx
	movl	$0, %eax
	subl	%edx, %eax
	addl	%eax, %eax
	addl	$1, %eax
	imull	-124(%rbp), %eax
	movl	-336(%rbp), %edx
	movslq	%edx, %rdx
	movq	%rdx, %rcx
	salq	$5, %rcx
	movq	-72(%rbp), %rdx
	addq	%rcx, %rdx
	cltq
	movq	%rax, 24(%rdx)
	addl	$1, -336(%rbp)
	movq	$44, -40(%rbp)
	jmp	.L32
.L21:
	movl	$0, %eax
	jmp	.L33
.L16:
	movl	-320(%rbp), %eax
	cmpl	-324(%rbp), %eax
	jge	.L34
	movq	$25, -40(%rbp)
	jmp	.L32
.L34:
	movq	$18, -40(%rbp)
	jmp	.L32
.L23:
	call	next_i
	movl	%eax, -216(%rbp)
	call	next_i
	movl	%eax, -212(%rbp)
	movl	-216(%rbp), %eax
	cltd
	shrl	$31, %edx
	addl	%edx, %eax
	andl	$1, %eax
	subl	%edx, %eax
	movl	%eax, %edx
	movl	$0, %eax
	subl	%edx, %eax
	addl	%eax, %eax
	addl	$1, %eax
	movl	-288(%rbp), %edx
	movslq	%edx, %rdx
	movq	%rdx, %rcx
	salq	$5, %rcx
	movq	-48(%rbp), %rdx
	addq	%rcx, %rdx
	imull	-212(%rbp), %eax
	movl	%eax, (%rdx)
	call	next_i
	movl	%eax, -208(%rbp)
	call	next_i
	movl	%eax, -204(%rbp)
	movl	-208(%rbp), %eax
	cltd
	shrl	$31, %edx
	addl	%edx, %eax
	andl	$1, %eax
	subl	%edx, %eax
	movl	%eax, %edx
	movl	$0, %eax
	subl	%edx, %eax
	addl	%eax, %eax
	addl	$1, %eax
	movl	-288(%rbp), %edx
	movslq	%edx, %rdx
	movq	%rdx, %rcx
	salq	$5, %rcx
	movq	-48(%rbp), %rdx
	addq	%rcx, %rdx
	imull	-204(%rbp), %eax
	movl	%eax, 4(%rdx)
	call	next_i
	movl	%eax, -200(%rbp)
	call	next_i
	movl	%eax, -196(%rbp)
	movl	-200(%rbp), %eax
	cltd
	shrl	$31, %edx
	addl	%edx, %eax
	andl	$1, %eax
	subl	%edx, %eax
	movl	%eax, %edx
	movl	$0, %eax
	subl	%edx, %eax
	addl	%eax, %eax
	addl	$1, %eax
	movl	-288(%rbp), %edx
	movslq	%edx, %rdx
	movq	%rdx, %rcx
	salq	$5, %rcx
	movq	-48(%rbp), %rdx
	addq	%rcx, %rdx
	imull	-196(%rbp), %eax
	movl	%eax, 8(%rdx)
	call	next_i
	movl	%eax, -192(%rbp)
	call	next_i
	movl	%eax, -188(%rbp)
	movl	-192(%rbp), %eax
	cltd
	shrl	$31, %edx
	addl	%edx, %eax
	andl	$1, %eax
	subl	%edx, %eax
	movl	%eax, %edx
	movl	$0, %eax
	subl	%edx, %eax
	addl	%eax, %eax
	addl	$1, %eax
	movl	-288(%rbp), %edx
	movslq	%edx, %rdx
	movq	%rdx, %rcx
	salq	$5, %rcx
	movq	-48(%rbp), %rdx
	addq	%rcx, %rdx
	imull	-188(%rbp), %eax
	movl	%eax, 12(%rdx)
	call	next_i
	movl	%eax, -184(%rbp)
	call	next_i
	movl	%eax, -180(%rbp)
	movl	-184(%rbp), %eax
	cltd
	shrl	$31, %edx
	addl	%edx, %eax
	andl	$1, %eax
	subl	%edx, %eax
	movl	%eax, %edx
	movl	$0, %eax
	subl	%edx, %eax
	addl	%eax, %eax
	addl	$1, %eax
	imull	-180(%rbp), %eax
	movl	-288(%rbp), %edx
	movslq	%edx, %rdx
	movq	%rdx, %rcx
	salq	$5, %rcx
	movq	-48(%rbp), %rdx
	addq	%rcx, %rdx
	cltq
	movq	%rax, 16(%rdx)
	call	next_i
	movl	%eax, -176(%rbp)
	call	next_i
	movl	%eax, -172(%rbp)
	movl	-176(%rbp), %eax
	cltd
	shrl	$31, %edx
	addl	%edx, %eax
	andl	$1, %eax
	subl	%edx, %eax
	movl	%eax, %edx
	movl	$0, %eax
	subl	%edx, %eax
	addl	%eax, %eax
	addl	$1, %eax
	imull	-172(%rbp), %eax
	movl	-288(%rbp), %edx
	movslq	%edx, %rdx
	movq	%rdx, %rcx
	salq	$5, %rcx
	movq	-48(%rbp), %rdx
	addq	%rcx, %rdx
	cltq
	movq	%rax, 24(%rdx)
	addl	$1, -288(%rbp)
	movq	$27, -40(%rbp)
	jmp	.L32
.L12:
	movl	$1, %eax
	jmp	.L33
.L28:
	movl	$10, -316(%rbp)
	movl	$10, -312(%rbp)
	movl	$100, -308(%rbp)
	movl	-308(%rbp), %eax
	cltq
	salq	$5, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -56(%rbp)
	movl	$0, -304(%rbp)
	movq	$41, -40(%rbp)
	jmp	.L32
.L19:
	movl	-288(%rbp), %eax
	cmpl	-292(%rbp), %eax
	jge	.L36
	movq	$17, -40(%rbp)
	jmp	.L32
.L36:
	movq	$43, -40(%rbp)
	jmp	.L32
.L14:
	movl	-312(%rbp), %edx
	movl	-316(%rbp), %ecx
	movq	-56(%rbp), %rax
	movl	%ecx, %esi
	movq	%rax, %rdi
	call	gsc_set_frame_size
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$19, -40(%rbp)
	jmp	.L32
.L18:
	call	next_i
	movl	%eax, -280(%rbp)
	call	next_i
	movl	%eax, -276(%rbp)
	movl	-280(%rbp), %eax
	cltd
	shrl	$31, %edx
	addl	%edx, %eax
	andl	$1, %eax
	subl	%edx, %eax
	movl	%eax, %edx
	movl	$0, %eax
	subl	%edx, %eax
	addl	%eax, %eax
	leal	1(%rax), %edx
	movl	-276(%rbp), %eax
	imull	%edx, %eax
	movl	%eax, -300(%rbp)
	call	next_i
	movl	%eax, -272(%rbp)
	call	next_i
	movl	%eax, -268(%rbp)
	movl	-272(%rbp), %eax
	cltd
	shrl	$31, %edx
	addl	%edx, %eax
	andl	$1, %eax
	subl	%edx, %eax
	movl	%eax, %edx
	movl	$0, %eax
	subl	%edx, %eax
	addl	%eax, %eax
	leal	1(%rax), %edx
	movl	-268(%rbp), %eax
	imull	%edx, %eax
	movl	%eax, -296(%rbp)
	movl	$1, -292(%rbp)
	movl	-292(%rbp), %eax
	cltq
	salq	$5, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, -48(%rbp)
	movl	$0, -288(%rbp)
	movq	$27, -40(%rbp)
	jmp	.L32
.L8:
	movl	-336(%rbp), %eax
	cmpl	-340(%rbp), %eax
	jge	.L38
	movq	$9, -40(%rbp)
	jmp	.L32
.L38:
	movq	$12, -40(%rbp)
	jmp	.L32
.L15:
	movl	$100, -348(%rbp)
	movl	$100, -344(%rbp)
	movl	$1, -340(%rbp)
	movl	-340(%rbp), %eax
	cltq
	salq	$5, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -32(%rbp)
	movq	-32(%rbp), %rax
	movq	%rax, -72(%rbp)
	movl	$0, -336(%rbp)
	movq	$44, -40(%rbp)
	jmp	.L32
.L11:
	movl	-304(%rbp), %eax
	cmpl	-308(%rbp), %eax
	jge	.L40
	movq	$4, -40(%rbp)
	jmp	.L32
.L40:
	movq	$34, -40(%rbp)
	jmp	.L32
.L13:
	cmpl	$3, -352(%rbp)
	je	.L42
	cmpl	$3, -352(%rbp)
	jg	.L43
	cmpl	$2, -352(%rbp)
	je	.L44
	cmpl	$2, -352(%rbp)
	jg	.L43
	cmpl	$0, -352(%rbp)
	je	.L45
	cmpl	$1, -352(%rbp)
	je	.L46
	jmp	.L43
.L42:
	movq	$28, -40(%rbp)
	jmp	.L47
.L44:
	movq	$6, -40(%rbp)
	jmp	.L47
.L46:
	movq	$3, -40(%rbp)
	jmp	.L47
.L45:
	movq	$33, -40(%rbp)
	jmp	.L47
.L43:
	movq	$11, -40(%rbp)
	nop
.L47:
	jmp	.L32
.L10:
	movl	-296(%rbp), %edx
	movl	-300(%rbp), %ecx
	movq	-48(%rbp), %rax
	movl	%ecx, %esi
	movq	%rax, %rdi
	call	gsc_set_frame_size
	movq	-48(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$19, -40(%rbp)
	jmp	.L32
.L31:
	cmpl	$2, -356(%rbp)
	je	.L48
	movq	$15, -40(%rbp)
	jmp	.L32
.L48:
	movq	$31, -40(%rbp)
	jmp	.L32
.L51:
	nop
.L32:
	jmp	.L50
.L33:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	main, .-main
	.globl	next_i
	.type	next_i, @function
next_i:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	$1, -8(%rbp)
.L58:
	cmpq	$2, -8(%rbp)
	je	.L53
	cmpq	$2, -8(%rbp)
	ja	.L60
	cmpq	$0, -8(%rbp)
	je	.L55
	cmpq	$1, -8(%rbp)
	jne	.L60
	movq	$0, -8(%rbp)
	jmp	.L56
.L55:
	movl	$0, -12(%rbp)
	addl	$1, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L56
.L53:
	movl	-12(%rbp), %ecx
	movslq	%ecx, %rax
	imulq	$1374389535, %rax, %rax
	shrq	$32, %rax
	sarl	$3, %eax
	movl	%ecx, %esi
	sarl	$31, %esi
	subl	%esi, %eax
	movl	%eax, %edx
	movl	%edx, %eax
	sall	$2, %eax
	addl	%edx, %eax
	leal	0(,%rax,4), %edx
	addl	%edx, %eax
	subl	%eax, %ecx
	movl	%ecx, %edx
	movslq	%edx, %rax
	leaq	0(,%rax,4), %rdx
	leaq	rand_primes(%rip), %rax
	movl	(%rdx,%rax), %eax
	jmp	.L59
.L60:
	nop
.L56:
	jmp	.L58
.L59:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	next_i, .-next_i
	.section	.rodata
	.align 8
.LC0:
	.string	"Usage:\n    prog [ARGS]\n\nARGS:\n       0            int-bounds\n       1            big-arr\n       2            big-arr-10x\n       3            empty\n"
	.text
	.globl	usage
	.type	usage, @function
usage:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$1, -8(%rbp)
.L66:
	cmpq	$0, -8(%rbp)
	je	.L67
	cmpq	$1, -8(%rbp)
	jne	.L68
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$0, -8(%rbp)
	jmp	.L64
.L68:
	nop
.L64:
	jmp	.L66
.L67:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	usage, .-usage
	.globl	gsc_set_frame_size
	.type	gsc_set_frame_size, @function
gsc_set_frame_size:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movl	%esi, -28(%rbp)
	movl	%edx, -32(%rbp)
	movq	$0, -8(%rbp)
.L75:
	cmpq	$2, -8(%rbp)
	je	.L76
	cmpq	$2, -8(%rbp)
	ja	.L77
	cmpq	$0, -8(%rbp)
	je	.L72
	cmpq	$1, -8(%rbp)
	jne	.L77
	movq	-24(%rbp), %rax
	movl	-28(%rbp), %edx
	movl	%edx, (%rax)
	movq	-24(%rbp), %rax
	movl	-32(%rbp), %edx
	movl	%edx, 4(%rax)
	movq	-24(%rbp), %rax
	movl	-28(%rbp), %edx
	movl	%edx, 8(%rax)
	movq	-24(%rbp), %rax
	movl	-32(%rbp), %edx
	movl	%edx, 12(%rax)
	movq	-24(%rbp), %rax
	movq	$0, 24(%rax)
	movq	-24(%rbp), %rax
	movq	$0, 16(%rax)
	movq	$2, -8(%rbp)
	jmp	.L73
.L72:
	movq	$1, -8(%rbp)
	jmp	.L73
.L77:
	nop
.L73:
	jmp	.L75
.L76:
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	gsc_set_frame_size, .-gsc_set_frame_size
	.globl	next_f
	.type	next_f, @function
next_f:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	$1, -8(%rbp)
.L86:
	cmpq	$2, -8(%rbp)
	je	.L79
	cmpq	$2, -8(%rbp)
	ja	.L88
	cmpq	$0, -8(%rbp)
	je	.L81
	cmpq	$1, -8(%rbp)
	jne	.L88
	movq	$2, -8(%rbp)
	jmp	.L82
.L81:
	movl	-12(%rbp), %ecx
	movslq	%ecx, %rax
	imulq	$1374389535, %rax, %rax
	shrq	$32, %rax
	sarl	$3, %eax
	movl	%ecx, %esi
	sarl	$31, %esi
	subl	%esi, %eax
	movl	%eax, %edx
	movl	%edx, %eax
	sall	$2, %eax
	addl	%edx, %eax
	leal	0(,%rax,4), %edx
	addl	%edx, %eax
	subl	%eax, %ecx
	movl	%ecx, %edx
	movslq	%edx, %rax
	leaq	0(,%rax,4), %rdx
	leaq	rand_primes(%rip), %rax
	movl	(%rdx,%rax), %eax
	movl	%eax, %eax
	testq	%rax, %rax
	js	.L83
	pxor	%xmm0, %xmm0
	cvtsi2ssq	%rax, %xmm0
	jmp	.L84
.L83:
	movq	%rax, %rdx
	shrq	%rdx
	andl	$1, %eax
	orq	%rax, %rdx
	pxor	%xmm0, %xmm0
	cvtsi2ssq	%rdx, %xmm0
	addss	%xmm0, %xmm0
.L84:
	movss	.LC1(%rip), %xmm1
	divss	%xmm1, %xmm0
	jmp	.L87
.L79:
	movl	$0, -12(%rbp)
	addl	$1, -12(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L82
.L88:
	nop
.L82:
	jmp	.L86
.L87:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	next_f, .-next_f
	.section	.rodata
	.align 4
.LC1:
	.long	1228472176
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

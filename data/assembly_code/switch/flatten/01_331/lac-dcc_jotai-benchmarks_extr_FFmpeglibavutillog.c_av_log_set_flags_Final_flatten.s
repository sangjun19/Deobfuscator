	.file	"lac-dcc_jotai-benchmarks_extr_FFmpeglibavutillog.c_av_log_set_flags_Final_flatten.c"
	.text
	.globl	_TIG_IZ_teBH_argc
	.bss
	.align 4
	.type	_TIG_IZ_teBH_argc, @object
	.size	_TIG_IZ_teBH_argc, 4
_TIG_IZ_teBH_argc:
	.zero	4
	.globl	flags
	.align 4
	.type	flags, @object
	.size	flags, 4
flags:
	.zero	4
	.globl	_TIG_IZ_teBH_envp
	.align 8
	.type	_TIG_IZ_teBH_envp, @object
	.size	_TIG_IZ_teBH_envp, 8
_TIG_IZ_teBH_envp:
	.zero	8
	.globl	rand_primes
	.align 32
	.type	rand_primes, @object
	.size	rand_primes, 100
rand_primes:
	.zero	100
	.globl	_TIG_IZ_teBH_argv
	.align 8
	.type	_TIG_IZ_teBH_argv, @object
	.size	_TIG_IZ_teBH_argv, 8
_TIG_IZ_teBH_argv:
	.zero	8
	.section	.rodata
	.align 8
.LC0:
	.string	"Usage:\n    prog [ARGS]\n\nARGS:\n       0            int-bounds\n       1            big-arr\n       2            big-arr-10x\n       3            empty\n"
	.text
	.globl	usage
	.type	usage, @function
usage:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$0, -8(%rbp)
.L6:
	cmpq	$0, -8(%rbp)
	je	.L2
	cmpq	$1, -8(%rbp)
	jne	.L8
	jmp	.L7
.L2:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$1, -8(%rbp)
	jmp	.L5
.L8:
	nop
.L5:
	jmp	.L6
.L7:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	usage, .-usage
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
	movq	$0, -8(%rbp)
.L15:
	cmpq	$2, -8(%rbp)
	je	.L10
	cmpq	$2, -8(%rbp)
	ja	.L17
	cmpq	$0, -8(%rbp)
	je	.L12
	cmpq	$1, -8(%rbp)
	jne	.L17
	movl	$0, -12(%rbp)
	addl	$1, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L13
.L12:
	movq	$1, -8(%rbp)
	jmp	.L13
.L10:
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
	jmp	.L16
.L17:
	nop
.L13:
	jmp	.L15
.L16:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	next_i, .-next_i
	.globl	av_log_set_flags
	.type	av_log_set_flags, @function
av_log_set_flags:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -20(%rbp)
	movq	$0, -8(%rbp)
.L23:
	cmpq	$0, -8(%rbp)
	je	.L19
	cmpq	$1, -8(%rbp)
	jne	.L25
	jmp	.L24
.L19:
	movl	-20(%rbp), %eax
	movl	%eax, flags(%rip)
	movq	$1, -8(%rbp)
	jmp	.L22
.L25:
	nop
.L22:
	jmp	.L23
.L24:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	av_log_set_flags, .-av_log_set_flags
	.globl	next_f
	.type	next_f, @function
next_f:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	$0, -8(%rbp)
.L34:
	cmpq	$2, -8(%rbp)
	je	.L27
	cmpq	$2, -8(%rbp)
	ja	.L36
	cmpq	$0, -8(%rbp)
	je	.L29
	cmpq	$1, -8(%rbp)
	jne	.L36
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
	js	.L30
	pxor	%xmm0, %xmm0
	cvtsi2ssq	%rax, %xmm0
	jmp	.L31
.L30:
	movq	%rax, %rdx
	shrq	%rdx
	andl	$1, %eax
	orq	%rax, %rdx
	pxor	%xmm0, %xmm0
	cvtsi2ssq	%rdx, %xmm0
	addss	%xmm0, %xmm0
.L31:
	movss	.LC1(%rip), %xmm1
	divss	%xmm1, %xmm0
	jmp	.L35
.L29:
	movq	$2, -8(%rbp)
	jmp	.L33
.L27:
	movl	$0, -12(%rbp)
	addl	$1, -12(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L33
.L36:
	nop
.L33:
	jmp	.L34
.L35:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	next_f, .-next_f
	.globl	main
	.type	main, @function
main:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$80, %rsp
	movl	%edi, -52(%rbp)
	movq	%rsi, -64(%rbp)
	movq	%rdx, -72(%rbp)
	movl	$0, flags(%rip)
	nop
.L38:
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
.L39:
	movq	$0, _TIG_IZ_teBH_envp(%rip)
	nop
.L40:
	movq	$0, _TIG_IZ_teBH_argv(%rip)
	nop
.L41:
	movl	$0, _TIG_IZ_teBH_argc(%rip)
	nop
	nop
.L42:
.L43:
#APP
# 169 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-teBH--0
# 0 "" 2
#NO_APP
	movl	-52(%rbp), %eax
	movl	%eax, _TIG_IZ_teBH_argc(%rip)
	movq	-64(%rbp), %rax
	movq	%rax, _TIG_IZ_teBH_argv(%rip)
	movq	-72(%rbp), %rax
	movq	%rax, _TIG_IZ_teBH_envp(%rip)
	nop
	movq	$10, -8(%rbp)
.L67:
	cmpq	$19, -8(%rbp)
	ja	.L68
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L46(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L46(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L46:
	.long	.L56-.L46
	.long	.L55-.L46
	.long	.L68-.L46
	.long	.L54-.L46
	.long	.L68-.L46
	.long	.L68-.L46
	.long	.L53-.L46
	.long	.L68-.L46
	.long	.L68-.L46
	.long	.L68-.L46
	.long	.L52-.L46
	.long	.L51-.L46
	.long	.L50-.L46
	.long	.L49-.L46
	.long	.L68-.L46
	.long	.L68-.L46
	.long	.L48-.L46
	.long	.L47-.L46
	.long	.L68-.L46
	.long	.L45-.L46
	.text
.L50:
	movl	$100, -28(%rbp)
	movl	-28(%rbp), %eax
	movl	%eax, %edi
	call	av_log_set_flags
	movq	$1, -8(%rbp)
	jmp	.L57
.L55:
	movl	$0, %eax
	jmp	.L58
.L54:
	call	next_i
	movl	%eax, -20(%rbp)
	call	next_i
	movl	%eax, -16(%rbp)
	movl	-20(%rbp), %eax
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
	movl	-16(%rbp), %eax
	imull	%edx, %eax
	movl	%eax, -12(%rbp)
	movl	-12(%rbp), %eax
	movl	%eax, %edi
	call	av_log_set_flags
	movq	$1, -8(%rbp)
	jmp	.L57
.L48:
	movq	-64(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movl	%eax, -36(%rbp)
	movl	-36(%rbp), %eax
	movl	%eax, -40(%rbp)
	movq	$19, -8(%rbp)
	jmp	.L57
.L51:
	movl	$255, -24(%rbp)
	movl	-24(%rbp), %eax
	movl	%eax, %edi
	call	av_log_set_flags
	movq	$1, -8(%rbp)
	jmp	.L57
.L49:
	movl	$10, -32(%rbp)
	movl	-32(%rbp), %eax
	movl	%eax, %edi
	call	av_log_set_flags
	movq	$1, -8(%rbp)
	jmp	.L57
.L45:
	cmpl	$3, -40(%rbp)
	je	.L59
	cmpl	$3, -40(%rbp)
	jg	.L60
	cmpl	$2, -40(%rbp)
	je	.L61
	cmpl	$2, -40(%rbp)
	jg	.L60
	cmpl	$0, -40(%rbp)
	je	.L62
	cmpl	$1, -40(%rbp)
	je	.L63
	jmp	.L60
.L59:
	movq	$3, -8(%rbp)
	jmp	.L64
.L61:
	movq	$13, -8(%rbp)
	jmp	.L64
.L63:
	movq	$11, -8(%rbp)
	jmp	.L64
.L62:
	movq	$12, -8(%rbp)
	jmp	.L64
.L60:
	movq	$17, -8(%rbp)
	nop
.L64:
	jmp	.L57
.L47:
	call	usage
	movq	$1, -8(%rbp)
	jmp	.L57
.L53:
	movl	$1, %eax
	jmp	.L58
.L52:
	cmpl	$2, -52(%rbp)
	je	.L65
	movq	$0, -8(%rbp)
	jmp	.L57
.L65:
	movq	$16, -8(%rbp)
	jmp	.L57
.L56:
	call	usage
	movq	$6, -8(%rbp)
	jmp	.L57
.L68:
	nop
.L57:
	jmp	.L67
.L58:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	main, .-main
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

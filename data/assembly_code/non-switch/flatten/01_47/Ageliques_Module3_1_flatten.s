	.file	"Ageliques_Module3_1_flatten.c"
	.text
	.globl	_TIG_IZ_ujyL_envp
	.bss
	.align 8
	.type	_TIG_IZ_ujyL_envp, @object
	.size	_TIG_IZ_ujyL_envp, 8
_TIG_IZ_ujyL_envp:
	.zero	8
	.globl	_TIG_IZ_ujyL_argv
	.align 8
	.type	_TIG_IZ_ujyL_argv, @object
	.size	_TIG_IZ_ujyL_argv, 8
_TIG_IZ_ujyL_argv:
	.zero	8
	.globl	_TIG_IZ_ujyL_argc
	.align 4
	.type	_TIG_IZ_ujyL_argc, @object
	.size	_TIG_IZ_ujyL_argc, 4
_TIG_IZ_ujyL_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"Side: %.2f, Square: %.2f\n"
	.text
	.globl	calculate_area
	.type	calculate_area, @function
calculate_area:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movss	%xmm0, -20(%rbp)
	movq	$1, -8(%rbp)
.L7:
	cmpq	$2, -8(%rbp)
	je	.L8
	cmpq	$2, -8(%rbp)
	ja	.L9
	cmpq	$0, -8(%rbp)
	je	.L4
	cmpq	$1, -8(%rbp)
	jne	.L9
	movq	$0, -8(%rbp)
	jmp	.L5
.L4:
	movss	-20(%rbp), %xmm0
	mulss	%xmm0, %xmm0
	movss	%xmm0, -12(%rbp)
	pxor	%xmm0, %xmm0
	cvtss2sd	-12(%rbp), %xmm0
	pxor	%xmm2, %xmm2
	cvtss2sd	-20(%rbp), %xmm2
	movq	%xmm2, %rax
	movapd	%xmm0, %xmm1
	movq	%rax, %xmm0
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$2, %eax
	call	printf@PLT
	movq	$2, -8(%rbp)
	jmp	.L5
.L9:
	nop
.L5:
	jmp	.L7
.L8:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	calculate_area, .-calculate_area
	.section	.rodata
.LC1:
	.string	"Error fork"
.LC2:
	.string	"Parent"
.LC3:
	.string	"Use: %s <side1> <side2> ...n"
.LC4:
	.string	"num_sides: %d \n"
.LC5:
	.string	"MID: %d \n"
.LC6:
	.string	"Doughter"
	.text
	.globl	main
	.type	main, @function
main:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$96, %rsp
	movl	%edi, -68(%rbp)
	movq	%rsi, -80(%rbp)
	movq	%rdx, -88(%rbp)
	movq	$0, _TIG_IZ_ujyL_envp(%rip)
	nop
.L11:
	movq	$0, _TIG_IZ_ujyL_argv(%rip)
	nop
.L12:
	movl	$0, _TIG_IZ_ujyL_argc(%rip)
	nop
	nop
.L13:
.L14:
#APP
# 116 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-ujyL--0
# 0 "" 2
#NO_APP
	movl	-68(%rbp), %eax
	movl	%eax, _TIG_IZ_ujyL_argc(%rip)
	movq	-80(%rbp), %rax
	movq	%rax, _TIG_IZ_ujyL_argv(%rip)
	movq	-88(%rbp), %rax
	movq	%rax, _TIG_IZ_ujyL_envp(%rip)
	nop
	movq	$13, -24(%rbp)
.L45:
	cmpq	$22, -24(%rbp)
	ja	.L46
	movq	-24(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L17(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L17(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L17:
	.long	.L32-.L17
	.long	.L46-.L17
	.long	.L46-.L17
	.long	.L31-.L17
	.long	.L30-.L17
	.long	.L29-.L17
	.long	.L28-.L17
	.long	.L27-.L17
	.long	.L26-.L17
	.long	.L25-.L17
	.long	.L46-.L17
	.long	.L24-.L17
	.long	.L46-.L17
	.long	.L23-.L17
	.long	.L22-.L17
	.long	.L21-.L17
	.long	.L46-.L17
	.long	.L46-.L17
	.long	.L20-.L17
	.long	.L19-.L17
	.long	.L18-.L17
	.long	.L46-.L17
	.long	.L16-.L17
	.text
.L20:
	movl	-40(%rbp), %eax
	cmpl	-52(%rbp), %eax
	jg	.L33
	movq	$8, -24(%rbp)
	jmp	.L35
.L33:
	movq	$19, -24(%rbp)
	jmp	.L35
.L30:
	movl	-52(%rbp), %eax
	movl	%eax, -44(%rbp)
	movq	$20, -24(%rbp)
	jmp	.L35
.L22:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$15, -24(%rbp)
	jmp	.L35
.L21:
	movl	$1, %eax
	jmp	.L36
.L26:
	movl	-40(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-80(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	atof@PLT
	movq	%xmm0, %rax
	movq	%rax, -8(%rbp)
	pxor	%xmm0, %xmm0
	cvtsd2ss	-8(%rbp), %xmm0
	movss	%xmm0, -28(%rbp)
	movl	-28(%rbp), %eax
	movd	%eax, %xmm0
	call	calculate_area
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	addl	$1, -40(%rbp)
	movq	$18, -24(%rbp)
	jmp	.L35
.L31:
	movq	-80(%rbp), %rax
	movq	(%rax), %rdx
	movq	stderr(%rip), %rax
	leaq	.LC3(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	$22, -24(%rbp)
	jmp	.L35
.L24:
	movl	-68(%rbp), %eax
	subl	$1, %eax
	movl	%eax, -56(%rbp)
	movl	-56(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-56(%rbp), %eax
	movl	%eax, %edx
	shrl	$31, %edx
	addl	%edx, %eax
	sarl	%eax
	movl	%eax, -52(%rbp)
	movl	-52(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	call	fork@PLT
	movl	%eax, -36(%rbp)
	movl	-36(%rbp), %eax
	movl	%eax, -48(%rbp)
	movq	$7, -24(%rbp)
	jmp	.L35
.L25:
	movl	-44(%rbp), %eax
	cltq
	addq	$1, %rax
	leaq	0(,%rax,8), %rdx
	movq	-80(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	atof@PLT
	movq	%xmm0, %rax
	movq	%rax, -16(%rbp)
	pxor	%xmm0, %xmm0
	cvtsd2ss	-16(%rbp), %xmm0
	movss	%xmm0, -32(%rbp)
	movl	-32(%rbp), %eax
	movd	%eax, %xmm0
	call	calculate_area
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	addl	$1, -44(%rbp)
	movq	$20, -24(%rbp)
	jmp	.L35
.L23:
	cmpl	$1, -68(%rbp)
	jg	.L37
	movq	$3, -24(%rbp)
	jmp	.L35
.L37:
	movq	$11, -24(%rbp)
	jmp	.L35
.L19:
	movl	$0, %eax
	jmp	.L36
.L28:
	movl	$1, -40(%rbp)
	movq	$18, -24(%rbp)
	jmp	.L35
.L16:
	movl	$1, %eax
	jmp	.L36
.L29:
	cmpl	$0, -48(%rbp)
	jne	.L39
	movq	$4, -24(%rbp)
	jmp	.L35
.L39:
	movq	$6, -24(%rbp)
	jmp	.L35
.L32:
	movl	$0, %edi
	call	exit@PLT
.L27:
	cmpl	$0, -48(%rbp)
	jns	.L41
	movq	$14, -24(%rbp)
	jmp	.L35
.L41:
	movq	$5, -24(%rbp)
	jmp	.L35
.L18:
	movl	-44(%rbp), %eax
	cmpl	-56(%rbp), %eax
	jge	.L43
	movq	$9, -24(%rbp)
	jmp	.L35
.L43:
	movq	$0, -24(%rbp)
	jmp	.L35
.L46:
	nop
.L35:
	jmp	.L45
.L36:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	main, .-main
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

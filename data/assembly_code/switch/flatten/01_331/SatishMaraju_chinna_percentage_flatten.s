	.file	"SatishMaraju_chinna_percentage_flatten.c"
	.text
	.globl	_TIG_IZ_qlnW_argv
	.bss
	.align 8
	.type	_TIG_IZ_qlnW_argv, @object
	.size	_TIG_IZ_qlnW_argv, 8
_TIG_IZ_qlnW_argv:
	.zero	8
	.globl	_TIG_IZ_qlnW_argc
	.align 4
	.type	_TIG_IZ_qlnW_argc, @object
	.size	_TIG_IZ_qlnW_argc, 4
_TIG_IZ_qlnW_argc:
	.zero	4
	.globl	_TIG_IZ_qlnW_envp
	.align 8
	.type	_TIG_IZ_qlnW_envp, @object
	.size	_TIG_IZ_qlnW_envp, 8
_TIG_IZ_qlnW_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"enter the 6 subjects marks:"
.LC1:
	.string	"%d%d%d%d%d%d"
.LC2:
	.string	"fail"
.LC3:
	.string	"honours"
.LC4:
	.string	"second division"
.LC5:
	.string	"third division"
.LC6:
	.string	"first division"
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
	subq	$80, %rsp
	movl	%edi, -52(%rbp)
	movq	%rsi, -64(%rbp)
	movq	%rdx, -72(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_qlnW_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_qlnW_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_qlnW_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 113 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-qlnW--0
# 0 "" 2
#NO_APP
	movl	-52(%rbp), %eax
	movl	%eax, _TIG_IZ_qlnW_argc(%rip)
	movq	-64(%rbp), %rax
	movq	%rax, _TIG_IZ_qlnW_argv(%rip)
	movq	-72(%rbp), %rax
	movq	%rax, _TIG_IZ_qlnW_envp(%rip)
	nop
	movq	$9, -16(%rbp)
.L28:
	cmpq	$13, -16(%rbp)
	ja	.L32
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L8(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L8(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L8:
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L32-.L8
	.long	.L14-.L8
	.long	.L32-.L8
	.long	.L32-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L32-.L8
	.long	.L7-.L8
	.text
.L14:
	cmpl	$100, -20(%rbp)
	jg	.L18
	cmpl	$80, -20(%rbp)
	jge	.L19
	cmpl	$79, -20(%rbp)
	jg	.L18
	cmpl	$60, -20(%rbp)
	jge	.L20
	cmpl	$59, -20(%rbp)
	jg	.L18
	cmpl	$50, -20(%rbp)
	jge	.L21
	cmpl	$39, -20(%rbp)
	jg	.L22
	cmpl	$0, -20(%rbp)
	jns	.L23
	jmp	.L18
.L22:
	movl	-20(%rbp), %eax
	subl	$40, %eax
	cmpl	$9, %eax
	ja	.L18
	jmp	.L30
.L23:
	movq	$11, -16(%rbp)
	jmp	.L25
.L30:
	movq	$7, -16(%rbp)
	jmp	.L25
.L21:
	movq	$10, -16(%rbp)
	jmp	.L25
.L20:
	movq	$2, -16(%rbp)
	jmp	.L25
.L19:
	movq	$13, -16(%rbp)
	jmp	.L25
.L18:
	movq	$0, -16(%rbp)
	nop
.L25:
	jmp	.L26
.L12:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-28(%rbp), %r8
	leaq	-32(%rbp), %rdi
	leaq	-36(%rbp), %rcx
	leaq	-40(%rbp), %rdx
	leaq	-44(%rbp), %rax
	subq	$8, %rsp
	leaq	-24(%rbp), %rsi
	pushq	%rsi
	movq	%r8, %r9
	movq	%rdi, %r8
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	addq	$16, %rsp
	movl	-44(%rbp), %edx
	movl	-40(%rbp), %eax
	addl	%eax, %edx
	movl	-36(%rbp), %eax
	addl	%eax, %edx
	movl	-32(%rbp), %eax
	addl	%eax, %edx
	movl	-28(%rbp), %eax
	addl	%eax, %edx
	movl	-24(%rbp), %eax
	addl	%edx, %eax
	movslq	%eax, %rdx
	imulq	$715827883, %rdx, %rdx
	shrq	$32, %rdx
	sarl	$31, %eax
	movl	%eax, %ecx
	movl	%edx, %eax
	subl	%ecx, %eax
	movl	%eax, -20(%rbp)
	movq	$4, -16(%rbp)
	jmp	.L26
.L16:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L29
	jmp	.L31
.L9:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -16(%rbp)
	jmp	.L26
.L11:
	movq	$8, -16(%rbp)
	jmp	.L26
.L7:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -16(%rbp)
	jmp	.L26
.L10:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -16(%rbp)
	jmp	.L26
.L17:
	movq	$1, -16(%rbp)
	jmp	.L26
.L13:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -16(%rbp)
	jmp	.L26
.L15:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -16(%rbp)
	jmp	.L26
.L32:
	nop
.L26:
	jmp	.L28
.L31:
	call	__stack_chk_fail@PLT
.L29:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
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

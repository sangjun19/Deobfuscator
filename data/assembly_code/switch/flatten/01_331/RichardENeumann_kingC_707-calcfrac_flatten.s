	.file	"RichardENeumann_kingC_707-calcfrac_flatten.c"
	.text
	.globl	_TIG_IZ_Dj8p_envp
	.bss
	.align 8
	.type	_TIG_IZ_Dj8p_envp, @object
	.size	_TIG_IZ_Dj8p_envp, 8
_TIG_IZ_Dj8p_envp:
	.zero	8
	.globl	_TIG_IZ_Dj8p_argc
	.align 4
	.type	_TIG_IZ_Dj8p_argc, @object
	.size	_TIG_IZ_Dj8p_argc, 4
_TIG_IZ_Dj8p_argc:
	.zero	4
	.globl	_TIG_IZ_Dj8p_argv
	.align 8
	.type	_TIG_IZ_Dj8p_argv, @object
	.size	_TIG_IZ_Dj8p_argv, 8
_TIG_IZ_Dj8p_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"The result is %d/%d\n"
	.align 8
.LC1:
	.string	"Enter the two fractions, separated by the operator [w/x(+-*/)y/z]: "
.LC2:
	.string	"%d/%d%1c%d/%d"
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
	movq	$0, _TIG_IZ_Dj8p_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_Dj8p_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_Dj8p_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 116 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-Dj8p--0
# 0 "" 2
#NO_APP
	movl	-52(%rbp), %eax
	movl	%eax, _TIG_IZ_Dj8p_argc(%rip)
	movq	-64(%rbp), %rax
	movq	%rax, _TIG_IZ_Dj8p_argv(%rip)
	movq	-72(%rbp), %rax
	movq	%rax, _TIG_IZ_Dj8p_envp(%rip)
	nop
	movq	$9, -16(%rbp)
.L26:
	cmpq	$17, -16(%rbp)
	ja	.L29
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
	.long	.L29-.L8
	.long	.L29-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L29-.L8
	.long	.L29-.L8
	.long	.L29-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L29-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L29-.L8
	.long	.L29-.L8
	.long	.L7-.L8
	.text
.L15:
	movq	$3, -16(%rbp)
	jmp	.L18
.L9:
	movzbl	-41(%rbp), %eax
	movsbl	%al, %eax
	cmpl	$47, %eax
	je	.L19
	cmpl	$47, %eax
	jg	.L20
	cmpl	$45, %eax
	je	.L21
	cmpl	$45, %eax
	jg	.L20
	cmpl	$42, %eax
	je	.L22
	cmpl	$43, %eax
	je	.L23
	jmp	.L20
.L22:
	movq	$13, -16(%rbp)
	jmp	.L24
.L19:
	movq	$8, -16(%rbp)
	jmp	.L24
.L21:
	movq	$10, -16(%rbp)
	jmp	.L24
.L23:
	movq	$0, -16(%rbp)
	jmp	.L24
.L20:
	movq	$4, -16(%rbp)
	nop
.L24:
	jmp	.L18
.L14:
	movl	-40(%rbp), %edx
	movl	-28(%rbp), %eax
	imull	%edx, %eax
	movl	%eax, -24(%rbp)
	movl	-36(%rbp), %edx
	movl	-32(%rbp), %eax
	imull	%edx, %eax
	movl	%eax, -20(%rbp)
	movq	$3, -16(%rbp)
	jmp	.L18
.L16:
	movl	-20(%rbp), %edx
	movl	-24(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$11, -16(%rbp)
	jmp	.L18
.L11:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L27
	jmp	.L28
.L13:
	movq	$17, -16(%rbp)
	jmp	.L18
.L10:
	movl	-40(%rbp), %edx
	movl	-32(%rbp), %eax
	imull	%edx, %eax
	movl	%eax, -24(%rbp)
	movl	-36(%rbp), %edx
	movl	-28(%rbp), %eax
	imull	%edx, %eax
	movl	%eax, -20(%rbp)
	movq	$3, -16(%rbp)
	jmp	.L18
.L7:
	movl	$0, -24(%rbp)
	movl	$0, -20(%rbp)
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-28(%rbp), %rdi
	leaq	-32(%rbp), %rsi
	leaq	-41(%rbp), %rcx
	leaq	-36(%rbp), %rdx
	leaq	-40(%rbp), %rax
	movq	%rdi, %r9
	movq	%rsi, %r8
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$14, -16(%rbp)
	jmp	.L18
.L12:
	movl	-40(%rbp), %edx
	movl	-28(%rbp), %eax
	imull	%edx, %eax
	movl	-32(%rbp), %ecx
	movl	-36(%rbp), %edx
	imull	%ecx, %edx
	subl	%edx, %eax
	movl	%eax, -24(%rbp)
	movl	-36(%rbp), %edx
	movl	-28(%rbp), %eax
	imull	%edx, %eax
	movl	%eax, -20(%rbp)
	movq	$3, -16(%rbp)
	jmp	.L18
.L17:
	movl	-40(%rbp), %edx
	movl	-28(%rbp), %eax
	movl	%edx, %ecx
	imull	%eax, %ecx
	movl	-32(%rbp), %edx
	movl	-36(%rbp), %eax
	imull	%edx, %eax
	addl	%ecx, %eax
	movl	%eax, -24(%rbp)
	movl	-36(%rbp), %edx
	movl	-28(%rbp), %eax
	imull	%edx, %eax
	movl	%eax, -20(%rbp)
	movq	$3, -16(%rbp)
	jmp	.L18
.L29:
	nop
.L18:
	jmp	.L26
.L28:
	call	__stack_chk_fail@PLT
.L27:
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

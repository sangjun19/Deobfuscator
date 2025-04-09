	.file	"AI-comp_Orientation2016Problems_main_flatten.c"
	.text
	.globl	_TIG_IZ_Oy0x_argv
	.bss
	.align 8
	.type	_TIG_IZ_Oy0x_argv, @object
	.size	_TIG_IZ_Oy0x_argv, 8
_TIG_IZ_Oy0x_argv:
	.zero	8
	.globl	_TIG_IZ_Oy0x_argc
	.align 4
	.type	_TIG_IZ_Oy0x_argc, @object
	.size	_TIG_IZ_Oy0x_argc, 4
_TIG_IZ_Oy0x_argc:
	.zero	4
	.globl	_TIG_IZ_Oy0x_envp
	.align 8
	.type	_TIG_IZ_Oy0x_envp, @object
	.size	_TIG_IZ_Oy0x_envp, 8
_TIG_IZ_Oy0x_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	" %d"
.LC1:
	.string	"%d\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB3:
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
	movq	$0, _TIG_IZ_Oy0x_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_Oy0x_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_Oy0x_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 103 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-Oy0x--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_Oy0x_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_Oy0x_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_Oy0x_envp(%rip)
	nop
	movq	$0, -16(%rbp)
.L25:
	cmpq	$12, -16(%rbp)
	ja	.L28
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
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L28-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L28-.L8
	.long	.L12-.L8
	.long	.L28-.L8
	.long	.L28-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L13:
	movl	-28(%rbp), %eax
	subl	$1896, %eax
	leal	3(%rax), %edx
	testl	%eax, %eax
	cmovs	%edx, %eax
	sarl	$2, %eax
	addl	$1, %eax
	movl	%eax, -20(%rbp)
	movq	$11, -16(%rbp)
	jmp	.L17
.L7:
	movl	-28(%rbp), %eax
	cmpl	$1895, %eax
	jle	.L18
	movq	$6, -16(%rbp)
	jmp	.L17
.L18:
	movq	$11, -16(%rbp)
	jmp	.L17
.L15:
	movl	-32(%rbp), %eax
	cmpl	%eax, -24(%rbp)
	jge	.L20
	movq	$10, -16(%rbp)
	jmp	.L17
.L20:
	movq	$9, -16(%rbp)
	jmp	.L17
.L14:
	leaq	-32(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	$0, -24(%rbp)
	movq	$1, -16(%rbp)
	jmp	.L17
.L9:
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -24(%rbp)
	movq	$1, -16(%rbp)
	jmp	.L17
.L11:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L26
	jmp	.L27
.L12:
	movl	-28(%rbp), %eax
	andl	$3, %eax
	testl	%eax, %eax
	jne	.L23
	movq	$4, -16(%rbp)
	jmp	.L17
.L23:
	movq	$11, -16(%rbp)
	jmp	.L17
.L10:
	leaq	-28(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	$-1, -20(%rbp)
	movq	$12, -16(%rbp)
	jmp	.L17
.L16:
	movq	$3, -16(%rbp)
	jmp	.L17
.L28:
	nop
.L17:
	jmp	.L25
.L27:
	call	__stack_chk_fail@PLT
.L26:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
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

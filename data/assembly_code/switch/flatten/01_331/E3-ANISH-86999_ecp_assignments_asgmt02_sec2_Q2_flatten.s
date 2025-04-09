	.file	"E3-ANISH-86999_ecp_assignments_asgmt02_sec2_Q2_flatten.c"
	.text
	.globl	_TIG_IZ_NJxV_argv
	.bss
	.align 8
	.type	_TIG_IZ_NJxV_argv, @object
	.size	_TIG_IZ_NJxV_argv, 8
_TIG_IZ_NJxV_argv:
	.zero	8
	.globl	_TIG_IZ_NJxV_argc
	.align 4
	.type	_TIG_IZ_NJxV_argc, @object
	.size	_TIG_IZ_NJxV_argc, 4
_TIG_IZ_NJxV_argc:
	.zero	4
	.globl	_TIG_IZ_NJxV_envp
	.align 8
	.type	_TIG_IZ_NJxV_envp, @object
	.size	_TIG_IZ_NJxV_envp, 8
_TIG_IZ_NJxV_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"mul = %d\n"
.LC1:
	.string	"sub = %d\n"
.LC2:
	.string	"Enter a number:"
.LC3:
	.string	"%d"
.LC4:
	.string	"Enter the operator:"
.LC5:
	.string	"%*c%c"
.LC6:
	.string	"Enter two values:"
.LC7:
	.string	"Invalid operator!"
.LC8:
	.string	"div = %d\n"
.LC9:
	.string	"add = %d\n"
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
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_NJxV_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_NJxV_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_NJxV_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 113 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-NJxV--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_NJxV_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_NJxV_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_NJxV_envp(%rip)
	nop
	movq	$9, -16(%rbp)
.L25:
	movq	-16(%rbp), %rax
	subq	$3, %rax
	cmpq	$11, %rax
	ja	.L28
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
	.long	.L14-.L8
	.long	.L28-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L28-.L8
	.long	.L10-.L8
	.long	.L28-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L15:
	movl	-28(%rbp), %edx
	movl	-24(%rbp), %eax
	imull	%edx, %eax
	movl	%eax, -20(%rbp)
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -16(%rbp)
	jmp	.L17
.L7:
	movl	-28(%rbp), %eax
	movl	-24(%rbp), %edx
	subl	%edx, %eax
	movl	%eax, -20(%rbp)
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -16(%rbp)
	jmp	.L17
.L12:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-28(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-29(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$13, -16(%rbp)
	jmp	.L17
.L16:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L26
	jmp	.L27
.L10:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$3, -16(%rbp)
	jmp	.L17
.L11:
	movq	$8, -16(%rbp)
	jmp	.L17
.L9:
	movzbl	-29(%rbp), %eax
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
.L19:
	movq	$5, -16(%rbp)
	jmp	.L24
.L22:
	movq	$4, -16(%rbp)
	jmp	.L24
.L21:
	movq	$14, -16(%rbp)
	jmp	.L24
.L23:
	movq	$7, -16(%rbp)
	jmp	.L24
.L20:
	movq	$11, -16(%rbp)
	nop
.L24:
	jmp	.L17
.L14:
	movl	-28(%rbp), %eax
	movl	-24(%rbp), %ecx
	cltd
	idivl	%ecx
	movl	%eax, -20(%rbp)
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -16(%rbp)
	jmp	.L17
.L13:
	movl	-28(%rbp), %edx
	movl	-24(%rbp), %eax
	addl	%edx, %eax
	movl	%eax, -20(%rbp)
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
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

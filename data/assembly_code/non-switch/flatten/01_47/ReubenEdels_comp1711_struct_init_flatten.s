	.file	"ReubenEdels_comp1711_struct_init_flatten.c"
	.text
	.globl	_TIG_IZ_8ypI_envp
	.bss
	.align 8
	.type	_TIG_IZ_8ypI_envp, @object
	.size	_TIG_IZ_8ypI_envp, 8
_TIG_IZ_8ypI_envp:
	.zero	8
	.globl	_TIG_IZ_8ypI_argc
	.align 4
	.type	_TIG_IZ_8ypI_argc, @object
	.size	_TIG_IZ_8ypI_argc, 4
_TIG_IZ_8ypI_argc:
	.zero	4
	.globl	_TIG_IZ_8ypI_argv
	.align 8
	.type	_TIG_IZ_8ypI_argv, @object
	.size	_TIG_IZ_8ypI_argv, 8
_TIG_IZ_8ypI_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"Input your name: "
.LC1:
	.string	"%s"
.LC2:
	.string	"Input your student id: "
.LC3:
	.string	"Input your mark: "
.LC4:
	.string	"%u"
.LC5:
	.string	"Student name: %s\n"
.LC6:
	.string	"Student ID:   %s\n"
.LC7:
	.string	"Final mark:   %u\n"
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
	subq	$96, %rsp
	movl	%edi, -68(%rbp)
	movq	%rsi, -80(%rbp)
	movq	%rdx, -88(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_8ypI_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_8ypI_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_8ypI_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 100 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-8ypI--0
# 0 "" 2
#NO_APP
	movl	-68(%rbp), %eax
	movl	%eax, _TIG_IZ_8ypI_argc(%rip)
	movq	-80(%rbp), %rax
	movq	%rax, _TIG_IZ_8ypI_argv(%rip)
	movq	-88(%rbp), %rax
	movq	%rax, _TIG_IZ_8ypI_envp(%rip)
	nop
	movq	$2, -56(%rbp)
.L11:
	cmpq	$2, -56(%rbp)
	je	.L6
	cmpq	$2, -56(%rbp)
	ja	.L14
	cmpq	$0, -56(%rbp)
	je	.L8
	cmpq	$1, -56(%rbp)
	jne	.L14
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-48(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-48(%rbp), %rax
	addq	$20, %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-48(%rbp), %rax
	addq	$32, %rax
	movq	%rax, %rsi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	-48(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-48(%rbp), %rax
	addq	$20, %rax
	movq	%rax, %rsi
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-16(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$0, -56(%rbp)
	jmp	.L9
.L8:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L12
	jmp	.L13
.L6:
	movq	$1, -56(%rbp)
	jmp	.L9
.L14:
	nop
.L9:
	jmp	.L11
.L13:
	call	__stack_chk_fail@PLT
.L12:
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

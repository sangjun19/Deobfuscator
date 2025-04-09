	.file	"amrutadahatonde10_c-code_c4_flatten.c"
	.text
	.globl	_TIG_IZ_B02T_argc
	.bss
	.align 4
	.type	_TIG_IZ_B02T_argc, @object
	.size	_TIG_IZ_B02T_argc, 4
_TIG_IZ_B02T_argc:
	.zero	4
	.globl	_TIG_IZ_B02T_argv
	.align 8
	.type	_TIG_IZ_B02T_argv, @object
	.size	_TIG_IZ_B02T_argv, 8
_TIG_IZ_B02T_argv:
	.zero	8
	.globl	_TIG_IZ_B02T_envp
	.align 8
	.type	_TIG_IZ_B02T_envp, @object
	.size	_TIG_IZ_B02T_envp, 8
_TIG_IZ_B02T_envp:
	.zero	8
	.section	.rodata
	.align 8
.LC0:
	.string	"The sum of the entered integers is: %d\n"
.LC1:
	.string	"Integer is: "
.LC2:
	.string	"%d"
.LC3:
	.string	"Enter 10 integers:"
	.text
	.globl	main
	.type	main, @function
main:
.LFB4:
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
	movq	$0, _TIG_IZ_B02T_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_B02T_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_B02T_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 126 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-B02T--0
# 0 "" 2
#NO_APP
	movl	-68(%rbp), %eax
	movl	%eax, _TIG_IZ_B02T_argc(%rip)
	movq	-80(%rbp), %rax
	movq	%rax, _TIG_IZ_B02T_argv(%rip)
	movq	-88(%rbp), %rax
	movq	%rax, _TIG_IZ_B02T_envp(%rip)
	nop
	movq	$3, -56(%rbp)
.L18:
	cmpq	$8, -56(%rbp)
	ja	.L21
	movq	-56(%rbp), %rax
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
	.long	.L13-.L8
	.long	.L21-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L21-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L21-.L8
	.long	.L7-.L8
	.text
.L7:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L19
	jmp	.L20
.L11:
	movq	$2, -56(%rbp)
	jmp	.L15
.L9:
	cmpl	$9, -60(%rbp)
	jg	.L16
	movq	$0, -56(%rbp)
	jmp	.L15
.L16:
	movq	$5, -56(%rbp)
	jmp	.L15
.L10:
	movl	-64(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$8, -56(%rbp)
	jmp	.L15
.L13:
	movl	-60(%rbp), %eax
	addl	$1, %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-48(%rbp), %rdx
	movl	-60(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-60(%rbp), %eax
	cltq
	movl	-48(%rbp,%rax,4), %eax
	addl	%eax, -64(%rbp)
	addl	$1, -60(%rbp)
	movq	$6, -56(%rbp)
	jmp	.L15
.L12:
	movl	$0, -64(%rbp)
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$0, -60(%rbp)
	movq	$6, -56(%rbp)
	jmp	.L15
.L21:
	nop
.L15:
	jmp	.L18
.L20:
	call	__stack_chk_fail@PLT
.L19:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
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

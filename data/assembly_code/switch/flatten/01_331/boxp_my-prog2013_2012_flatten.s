	.file	"boxp_my-prog2013_2012_flatten.c"
	.text
	.globl	_TIG_IZ_2NcU_argv
	.bss
	.align 8
	.type	_TIG_IZ_2NcU_argv, @object
	.size	_TIG_IZ_2NcU_argv, 8
_TIG_IZ_2NcU_argv:
	.zero	8
	.globl	_TIG_IZ_2NcU_envp
	.align 8
	.type	_TIG_IZ_2NcU_envp, @object
	.size	_TIG_IZ_2NcU_envp, 8
_TIG_IZ_2NcU_envp:
	.zero	8
	.globl	_TIG_IZ_2NcU_argc
	.align 4
	.type	_TIG_IZ_2NcU_argc, @object
	.size	_TIG_IZ_2NcU_argc, 4
_TIG_IZ_2NcU_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"%c"
.LC1:
	.string	"radian"
.LC2:
	.string	"degree"
.LC3:
	.string	"Illegal"
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
	movq	$0, _TIG_IZ_2NcU_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_2NcU_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_2NcU_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 107 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-2NcU--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_2NcU_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_2NcU_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_2NcU_envp(%rip)
	nop
	movq	$1, -16(%rbp)
.L19:
	cmpq	$8, -16(%rbp)
	ja	.L22
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
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L22-.L8
	.long	.L22-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L22-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L11:
	movzbl	-17(%rbp), %eax
	movsbl	%al, %eax
	cmpl	$100, %eax
	je	.L14
	cmpl	$114, %eax
	jne	.L15
	movq	$5, -16(%rbp)
	jmp	.L16
.L14:
	movq	$0, -16(%rbp)
	jmp	.L16
.L15:
	movq	$7, -16(%rbp)
	nop
.L16:
	jmp	.L17
.L7:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L20
	jmp	.L21
.L12:
	leaq	-17(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$4, -16(%rbp)
	jmp	.L17
.L10:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$8, -16(%rbp)
	jmp	.L17
.L13:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$8, -16(%rbp)
	jmp	.L17
.L9:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$8, -16(%rbp)
	jmp	.L17
.L22:
	nop
.L17:
	jmp	.L19
.L21:
	call	__stack_chk_fail@PLT
.L20:
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

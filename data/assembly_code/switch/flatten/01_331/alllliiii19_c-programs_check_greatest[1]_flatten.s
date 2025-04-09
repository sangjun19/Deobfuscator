	.file	"alllliiii19_c-programs_check_greatest[1]_flatten.c"
	.text
	.globl	_TIG_IZ_KYTs_envp
	.bss
	.align 8
	.type	_TIG_IZ_KYTs_envp, @object
	.size	_TIG_IZ_KYTs_envp, 8
_TIG_IZ_KYTs_envp:
	.zero	8
	.globl	_TIG_IZ_KYTs_argc
	.align 4
	.type	_TIG_IZ_KYTs_argc, @object
	.size	_TIG_IZ_KYTs_argc, 4
_TIG_IZ_KYTs_argc:
	.zero	4
	.globl	_TIG_IZ_KYTs_argv
	.align 8
	.type	_TIG_IZ_KYTs_argv, @object
	.size	_TIG_IZ_KYTs_argv, 8
_TIG_IZ_KYTs_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"%x is greater"
.LC1:
	.string	"Enter the numbers"
.LC2:
	.string	"%d%d"
.LC3:
	.string	"%y is greater"
	.text
	.globl	main
	.type	main, @function
main:
.LFB2:
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
	movq	$0, _TIG_IZ_KYTs_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_KYTs_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_KYTs_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 86 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-KYTs--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_KYTs_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_KYTs_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_KYTs_envp(%rip)
	nop
	movq	$6, -16(%rbp)
.L26:
	cmpq	$10, -16(%rbp)
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
	.long	.L29-.L8
	.long	.L29-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L30-.L8
	.text
.L14:
	movq	$10, -16(%rbp)
	jmp	.L17
.L10:
	movl	-24(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$10, -16(%rbp)
	jmp	.L17
.L15:
	movl	-24(%rbp), %edx
	movl	-20(%rbp), %eax
	cmpl	%eax, %edx
	je	.L18
	movq	$7, -16(%rbp)
	jmp	.L17
.L18:
	movq	$10, -16(%rbp)
	jmp	.L17
.L9:
	movl	-24(%rbp), %edx
	movl	-20(%rbp), %eax
	cmpl	%eax, %edx
	je	.L20
	movq	$8, -16(%rbp)
	jmp	.L17
.L20:
	movq	$10, -16(%rbp)
	jmp	.L17
.L12:
	movq	$5, -16(%rbp)
	jmp	.L17
.L13:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-20(%rbp), %rdx
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$2, -16(%rbp)
	jmp	.L17
.L11:
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$10, -16(%rbp)
	jmp	.L17
.L16:
	movl	-24(%rbp), %edx
	movl	-20(%rbp), %eax
	cmpl	%eax, %edx
	setl	%al
	movzbl	%al, %eax
	testl	%eax, %eax
	je	.L23
	cmpl	$1, %eax
	jne	.L24
	movq	$3, -16(%rbp)
	jmp	.L25
.L23:
	movq	$9, -16(%rbp)
	jmp	.L25
.L24:
	movq	$4, -16(%rbp)
	nop
.L25:
	jmp	.L17
.L29:
	nop
.L17:
	jmp	.L26
.L30:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L28
	call	__stack_chk_fail@PLT
.L28:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
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

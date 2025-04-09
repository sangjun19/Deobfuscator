	.file	"mohitsharma021_assignment-__flatten.c"
	.text
	.globl	_TIG_IZ_jL3S_envp
	.bss
	.align 8
	.type	_TIG_IZ_jL3S_envp, @object
	.size	_TIG_IZ_jL3S_envp, 8
_TIG_IZ_jL3S_envp:
	.zero	8
	.globl	_TIG_IZ_jL3S_argv
	.align 8
	.type	_TIG_IZ_jL3S_argv, @object
	.size	_TIG_IZ_jL3S_argv, 8
_TIG_IZ_jL3S_argv:
	.zero	8
	.globl	_TIG_IZ_jL3S_argc
	.align 4
	.type	_TIG_IZ_jL3S_argc, @object
	.size	_TIG_IZ_jL3S_argc, 4
_TIG_IZ_jL3S_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"%d "
.LC1:
	.string	"Merged array: "
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
	subq	$160, %rsp
	movl	%edi, -132(%rbp)
	movq	%rsi, -144(%rbp)
	movq	%rdx, -152(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_jL3S_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_jL3S_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_jL3S_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 106 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-jL3S--0
# 0 "" 2
#NO_APP
	movl	-132(%rbp), %eax
	movl	%eax, _TIG_IZ_jL3S_argc(%rip)
	movq	-144(%rbp), %rax
	movq	%rax, _TIG_IZ_jL3S_argv(%rip)
	movq	-152(%rbp), %rax
	movq	%rax, _TIG_IZ_jL3S_envp(%rip)
	nop
	movq	$9, -104(%rbp)
.L28:
	cmpq	$20, -104(%rbp)
	ja	.L31
	movq	-104(%rbp), %rax
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
	.long	.L31-.L8
	.long	.L19-.L8
	.long	.L31-.L8
	.long	.L31-.L8
	.long	.L31-.L8
	.long	.L31-.L8
	.long	.L18-.L8
	.long	.L31-.L8
	.long	.L31-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L31-.L8
	.long	.L9-.L8
	.long	.L31-.L8
	.long	.L7-.L8
	.text
.L9:
	movl	$10, -64(%rbp)
	movl	$12, -60(%rbp)
	movl	$18, -56(%rbp)
	movl	$22, -52(%rbp)
	movl	$26, -48(%rbp)
	movl	$39, -44(%rbp)
	movl	$36, -40(%rbp)
	movl	$41, -36(%rbp)
	movl	$45, -32(%rbp)
	movl	$9, -124(%rbp)
	movq	$11, -104(%rbp)
	jmp	.L20
.L12:
	movl	-108(%rbp), %eax
	cltq
	movl	-64(%rbp,%rax,4), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -108(%rbp)
	movq	$20, -104(%rbp)
	jmp	.L20
.L11:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L29
	jmp	.L30
.L14:
	movl	$10, %edi
	call	putchar@PLT
	movq	$15, -104(%rbp)
	jmp	.L20
.L19:
	movl	$9, -120(%rbp)
	movl	$1, -96(%rbp)
	movl	$2, -92(%rbp)
	movl	$3, -88(%rbp)
	movl	$4, -84(%rbp)
	movl	$5, -80(%rbp)
	movl	$5, -116(%rbp)
	movl	$0, -112(%rbp)
	movq	$13, -104(%rbp)
	jmp	.L20
.L10:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$0, -108(%rbp)
	movq	$20, -104(%rbp)
	jmp	.L20
.L15:
	cmpl	$13, -124(%rbp)
	jbe	.L22
	movq	$1, -104(%rbp)
	jmp	.L20
.L22:
	movq	$6, -104(%rbp)
	jmp	.L20
.L17:
	movq	$18, -104(%rbp)
	jmp	.L20
.L13:
	movl	-112(%rbp), %eax
	cmpl	-116(%rbp), %eax
	jge	.L24
	movq	$10, -104(%rbp)
	jmp	.L20
.L24:
	movq	$16, -104(%rbp)
	jmp	.L20
.L18:
	movl	-124(%rbp), %eax
	movl	$0, -64(%rbp,%rax,4)
	addl	$1, -124(%rbp)
	movq	$11, -104(%rbp)
	jmp	.L20
.L16:
	movl	-120(%rbp), %edx
	movl	-112(%rbp), %eax
	leal	(%rdx,%rax), %ecx
	movl	-112(%rbp), %eax
	cltq
	movl	-96(%rbp,%rax,4), %edx
	movslq	%ecx, %rax
	movl	%edx, -64(%rbp,%rax,4)
	addl	$1, -112(%rbp)
	movq	$13, -104(%rbp)
	jmp	.L20
.L7:
	movl	-120(%rbp), %edx
	movl	-116(%rbp), %eax
	addl	%edx, %eax
	cmpl	%eax, -108(%rbp)
	jge	.L26
	movq	$14, -104(%rbp)
	jmp	.L20
.L26:
	movq	$12, -104(%rbp)
	jmp	.L20
.L31:
	nop
.L20:
	jmp	.L28
.L30:
	call	__stack_chk_fail@PLT
.L29:
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

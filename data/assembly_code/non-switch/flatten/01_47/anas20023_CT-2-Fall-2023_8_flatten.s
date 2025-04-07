	.file	"anas20023_CT-2-Fall-2023_8_flatten.c"
	.text
	.globl	_TIG_IZ_AoZy_argv
	.bss
	.align 8
	.type	_TIG_IZ_AoZy_argv, @object
	.size	_TIG_IZ_AoZy_argv, 8
_TIG_IZ_AoZy_argv:
	.zero	8
	.globl	_TIG_IZ_AoZy_envp
	.align 8
	.type	_TIG_IZ_AoZy_envp, @object
	.size	_TIG_IZ_AoZy_envp, 8
_TIG_IZ_AoZy_envp:
	.zero	8
	.globl	_TIG_IZ_AoZy_argc
	.align 4
	.type	_TIG_IZ_AoZy_argc, @object
	.size	_TIG_IZ_AoZy_argc, 4
_TIG_IZ_AoZy_argc:
	.zero	4
	.text
	.globl	Count_len
	.type	Count_len, @function
Count_len:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movq	$7, -8(%rbp)
.L13:
	cmpq	$7, -8(%rbp)
	ja	.L15
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L4(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L4(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L4:
	.long	.L8-.L4
	.long	.L15-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L15-.L4
	.long	.L15-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L6:
	movl	-16(%rbp), %eax
	movslq	%eax, %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	je	.L9
	movq	$6, -8(%rbp)
	jmp	.L11
.L9:
	movq	$2, -8(%rbp)
	jmp	.L11
.L5:
	addl	$1, -12(%rbp)
	addl	$1, -16(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L11
.L8:
	movl	$0, -12(%rbp)
	movl	$0, -16(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L11
.L3:
	movq	$0, -8(%rbp)
	jmp	.L11
.L7:
	movl	-12(%rbp), %eax
	jmp	.L14
.L15:
	nop
.L11:
	jmp	.L13
.L14:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	Count_len, .-Count_len
	.section	.rodata
.LC0:
	.string	"Reversed String =%s\n"
.LC1:
	.string	"Enter String:"
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
	subq	$688, %rsp
	movl	%edi, -660(%rbp)
	movq	%rsi, -672(%rbp)
	movq	%rdx, -680(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_AoZy_envp(%rip)
	nop
.L17:
	movq	$0, _TIG_IZ_AoZy_argv(%rip)
	nop
.L18:
	movl	$0, _TIG_IZ_AoZy_argc(%rip)
	nop
	nop
.L19:
.L20:
#APP
# 137 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-AoZy--0
# 0 "" 2
#NO_APP
	movl	-660(%rbp), %eax
	movl	%eax, _TIG_IZ_AoZy_argc(%rip)
	movq	-672(%rbp), %rax
	movq	%rax, _TIG_IZ_AoZy_argv(%rip)
	movq	-680(%rbp), %rax
	movq	%rax, _TIG_IZ_AoZy_envp(%rip)
	nop
	movq	$8, -632(%rbp)
.L33:
	cmpq	$8, -632(%rbp)
	ja	.L36
	movq	-632(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L23(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L23(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L23:
	.long	.L28-.L23
	.long	.L36-.L23
	.long	.L27-.L23
	.long	.L26-.L23
	.long	.L36-.L23
	.long	.L36-.L23
	.long	.L25-.L23
	.long	.L24-.L23
	.long	.L22-.L23
	.text
.L22:
	movq	$2, -632(%rbp)
	jmp	.L29
.L26:
	movl	-648(%rbp), %eax
	subl	$1, %eax
	cltq
	movzbl	-624(%rbp,%rax), %edx
	movl	-644(%rbp), %eax
	cltq
	movb	%dl, -320(%rbp,%rax)
	addl	$1, -644(%rbp)
	subl	$1, -648(%rbp)
	movq	$0, -632(%rbp)
	jmp	.L29
.L25:
	movl	-644(%rbp), %eax
	cltq
	movb	$0, -320(%rbp,%rax)
	leaq	-320(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$7, -632(%rbp)
	jmp	.L29
.L28:
	movl	-644(%rbp), %eax
	cmpl	-640(%rbp), %eax
	jge	.L30
	movq	$3, -632(%rbp)
	jmp	.L29
.L30:
	movq	$6, -632(%rbp)
	jmp	.L29
.L24:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L34
	jmp	.L35
.L27:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-624(%rbp), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	gets@PLT
	leaq	-624(%rbp), %rax
	movq	%rax, %rdi
	call	Count_len
	movl	%eax, -636(%rbp)
	movl	-636(%rbp), %eax
	movl	%eax, -640(%rbp)
	movl	-640(%rbp), %eax
	movl	%eax, -648(%rbp)
	movl	$0, -644(%rbp)
	movq	$0, -632(%rbp)
	jmp	.L29
.L36:
	nop
.L29:
	jmp	.L33
.L35:
	call	__stack_chk_fail@PLT
.L34:
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

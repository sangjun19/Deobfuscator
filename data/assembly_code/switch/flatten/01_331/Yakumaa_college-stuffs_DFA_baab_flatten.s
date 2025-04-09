	.file	"Yakumaa_college-stuffs_DFA_baab_flatten.c"
	.text
	.globl	_TIG_IZ_spGP_envp
	.bss
	.align 8
	.type	_TIG_IZ_spGP_envp, @object
	.size	_TIG_IZ_spGP_envp, 8
_TIG_IZ_spGP_envp:
	.zero	8
	.globl	_TIG_IZ_spGP_argv
	.align 8
	.type	_TIG_IZ_spGP_argv, @object
	.size	_TIG_IZ_spGP_argv, 8
_TIG_IZ_spGP_argv:
	.zero	8
	.globl	_TIG_IZ_spGP_argc
	.align 4
	.type	_TIG_IZ_spGP_argc, @object
	.size	_TIG_IZ_spGP_argc, 4
_TIG_IZ_spGP_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"The string is accepted"
.LC1:
	.string	"Enter a string: "
.LC2:
	.string	"%s"
.LC3:
	.string	"The string is invalid"
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
	subq	$80, %rsp
	movl	%edi, -52(%rbp)
	movq	%rsi, -64(%rbp)
	movq	%rdx, -72(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_spGP_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_spGP_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_spGP_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 105 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-spGP--0
# 0 "" 2
#NO_APP
	movl	-52(%rbp), %eax
	movl	%eax, _TIG_IZ_spGP_argc(%rip)
	movq	-64(%rbp), %rax
	movq	%rax, _TIG_IZ_spGP_argv(%rip)
	movq	-72(%rbp), %rax
	movq	%rax, _TIG_IZ_spGP_envp(%rip)
	nop
	movq	$17, -40(%rbp)
.L52:
	cmpq	$26, -40(%rbp)
	ja	.L55
	movq	-40(%rbp), %rax
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
	.long	.L28-.L8
	.long	.L27-.L8
	.long	.L26-.L8
	.long	.L25-.L8
	.long	.L24-.L8
	.long	.L23-.L8
	.long	.L22-.L8
	.long	.L21-.L8
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L55-.L8
	.long	.L55-.L8
	.long	.L18-.L8
	.long	.L55-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L55-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L55-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L9:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L53
	jmp	.L54
.L25:
	movl	-48(%rbp), %eax
	cltq
	movzbl	-32(%rbp,%rax), %eax
	cmpb	$97, %al
	jne	.L31
	movq	$8, -40(%rbp)
	jmp	.L33
.L31:
	movq	$0, -40(%rbp)
	jmp	.L33
.L17:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$25, -40(%rbp)
	jmp	.L33
.L21:
	movl	$2, -44(%rbp)
	movq	$16, -40(%rbp)
	jmp	.L33
.L28:
	movl	-48(%rbp), %eax
	cltq
	movzbl	-32(%rbp,%rax), %eax
	cmpb	$98, %al
	jne	.L34
	movq	$3, -40(%rbp)
	jmp	.L33
.L34:
	movq	$6, -40(%rbp)
	jmp	.L33
.L11:
	movl	-48(%rbp), %eax
	cltq
	movzbl	-32(%rbp,%rax), %eax
	cmpb	$97, %al
	jne	.L36
	movq	$2, -40(%rbp)
	jmp	.L33
.L36:
	movq	$10, -40(%rbp)
	jmp	.L33
.L26:
	movl	$3, -44(%rbp)
	movq	$16, -40(%rbp)
	jmp	.L33
.L16:
	addl	$1, -48(%rbp)
	movq	$21, -40(%rbp)
	jmp	.L33
.L10:
	cmpl	$4, -44(%rbp)
	ja	.L38
	movl	-44(%rbp), %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L40(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L40(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L40:
	.long	.L44-.L40
	.long	.L43-.L40
	.long	.L42-.L40
	.long	.L41-.L40
	.long	.L39-.L40
	.text
.L39:
	movq	$16, -40(%rbp)
	jmp	.L45
.L41:
	movq	$1, -40(%rbp)
	jmp	.L45
.L42:
	movq	$23, -40(%rbp)
	jmp	.L45
.L43:
	movq	$4, -40(%rbp)
	jmp	.L45
.L44:
	movq	$9, -40(%rbp)
	jmp	.L45
.L38:
	movq	$13, -40(%rbp)
	nop
.L45:
	jmp	.L33
.L12:
	movl	-48(%rbp), %eax
	cltq
	movzbl	-32(%rbp,%rax), %eax
	testb	%al, %al
	je	.L46
	movq	$24, -40(%rbp)
	jmp	.L33
.L46:
	movq	$5, -40(%rbp)
	jmp	.L33
.L7:
	movl	$1, -44(%rbp)
	movq	$16, -40(%rbp)
	jmp	.L33
.L20:
	movl	-48(%rbp), %eax
	cltq
	movzbl	-32(%rbp,%rax), %eax
	cmpb	$98, %al
	jne	.L48
	movq	$26, -40(%rbp)
	jmp	.L33
.L48:
	movq	$20, -40(%rbp)
	jmp	.L33
.L18:
	movq	$16, -40(%rbp)
	jmp	.L33
.L14:
	movl	$0, -48(%rbp)
	movl	$0, -44(%rbp)
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-32(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$21, -40(%rbp)
	jmp	.L33
.L15:
	movq	$19, -40(%rbp)
	jmp	.L33
.L23:
	movl	$4, -44(%rbp)
	movq	$16, -40(%rbp)
	jmp	.L33
.L24:
	cmpl	$3, -44(%rbp)
	jne	.L50
	movq	$15, -40(%rbp)
	jmp	.L33
.L50:
	movq	$7, -40(%rbp)
	jmp	.L33
.L19:
	movl	$4, -44(%rbp)
	movq	$16, -40(%rbp)
	jmp	.L33
.L29:
	movl	$4, -44(%rbp)
	movq	$16, -40(%rbp)
	jmp	.L33
.L22:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$25, -40(%rbp)
	jmp	.L33
.L27:
	movl	$3, -44(%rbp)
	movq	$16, -40(%rbp)
	jmp	.L33
.L13:
	movl	$4, -44(%rbp)
	movq	$16, -40(%rbp)
	jmp	.L33
.L55:
	nop
.L33:
	jmp	.L52
.L54:
	call	__stack_chk_fail@PLT
.L53:
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

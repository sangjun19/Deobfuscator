	.file	"TomasZur_C-Programming-a-modern-approach_project6_flatten.c"
	.text
	.globl	_TIG_IZ_CqrG_argv
	.bss
	.align 8
	.type	_TIG_IZ_CqrG_argv, @object
	.size	_TIG_IZ_CqrG_argv, 8
_TIG_IZ_CqrG_argv:
	.zero	8
	.globl	_TIG_IZ_CqrG_envp
	.align 8
	.type	_TIG_IZ_CqrG_envp, @object
	.size	_TIG_IZ_CqrG_envp, 8
_TIG_IZ_CqrG_envp:
	.zero	8
	.globl	_TIG_IZ_CqrG_argc
	.align 4
	.type	_TIG_IZ_CqrG_argc, @object
	.size	_TIG_IZ_CqrG_argc, 4
_TIG_IZ_CqrG_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"!!!!!!!!!!"
.LC1:
	.string	"Enter a message: "
.LC2:
	.string	"Message: "
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
	subq	$176, %rsp
	movl	%edi, -148(%rbp)
	movq	%rsi, -160(%rbp)
	movq	%rdx, -168(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_CqrG_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_CqrG_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_CqrG_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 170 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-CqrG--0
# 0 "" 2
#NO_APP
	movl	-148(%rbp), %eax
	movl	%eax, _TIG_IZ_CqrG_argc(%rip)
	movq	-160(%rbp), %rax
	movq	%rax, _TIG_IZ_CqrG_argv(%rip)
	movq	-168(%rbp), %rax
	movq	%rax, _TIG_IZ_CqrG_envp(%rip)
	nop
	movq	$22, -120(%rbp)
.L42:
	cmpq	$32, -120(%rbp)
	ja	.L45
	movq	-120(%rbp), %rax
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
	.long	.L26-.L8
	.long	.L45-.L8
	.long	.L45-.L8
	.long	.L25-.L8
	.long	.L24-.L8
	.long	.L23-.L8
	.long	.L22-.L8
	.long	.L21-.L8
	.long	.L20-.L8
	.long	.L45-.L8
	.long	.L19-.L8
	.long	.L45-.L8
	.long	.L18-.L8
	.long	.L45-.L8
	.long	.L17-.L8
	.long	.L45-.L8
	.long	.L45-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L45-.L8
	.long	.L45-.L8
	.long	.L13-.L8
	.long	.L45-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L45-.L8
	.long	.L10-.L8
	.long	.L45-.L8
	.long	.L9-.L8
	.long	.L45-.L8
	.long	.L45-.L8
	.long	.L7-.L8
	.text
.L15:
	movl	-132(%rbp), %eax
	cltq
	movb	$49, -112(%rbp,%rax)
	movq	$6, -120(%rbp)
	jmp	.L27
.L11:
	movl	-136(%rbp), %eax
	cltq
	movzbl	-137(%rbp), %edx
	movb	%dl, -112(%rbp,%rax)
	addl	$1, -136(%rbp)
	movq	$32, -120(%rbp)
	jmp	.L27
.L24:
	movl	-132(%rbp), %eax
	cltq
	movb	$56, -112(%rbp,%rax)
	movq	$6, -120(%rbp)
	jmp	.L27
.L17:
	movl	-132(%rbp), %eax
	cltq
	movzbl	-112(%rbp,%rax), %eax
	movsbl	%al, %eax
	subl	$65, %eax
	cmpl	$18, %eax
	ja	.L28
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L30(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L30(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L30:
	.long	.L35-.L30
	.long	.L34-.L30
	.long	.L28-.L30
	.long	.L28-.L30
	.long	.L33-.L30
	.long	.L28-.L30
	.long	.L28-.L30
	.long	.L28-.L30
	.long	.L32-.L30
	.long	.L28-.L30
	.long	.L28-.L30
	.long	.L28-.L30
	.long	.L28-.L30
	.long	.L28-.L30
	.long	.L31-.L30
	.long	.L28-.L30
	.long	.L28-.L30
	.long	.L28-.L30
	.long	.L29-.L30
	.text
.L29:
	movq	$10, -120(%rbp)
	jmp	.L36
.L31:
	movq	$8, -120(%rbp)
	jmp	.L36
.L32:
	movq	$18, -120(%rbp)
	jmp	.L36
.L33:
	movq	$27, -120(%rbp)
	jmp	.L36
.L34:
	movq	$4, -120(%rbp)
	jmp	.L36
.L35:
	movq	$19, -120(%rbp)
	jmp	.L36
.L28:
	movq	$17, -120(%rbp)
	nop
.L36:
	jmp	.L27
.L18:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$7, -120(%rbp)
	jmp	.L27
.L20:
	movl	-132(%rbp), %eax
	cltq
	movb	$48, -112(%rbp,%rax)
	movq	$6, -120(%rbp)
	jmp	.L27
.L25:
	movl	-132(%rbp), %eax
	cltq
	movzbl	-112(%rbp,%rax), %eax
	movsbl	%al, %eax
	movl	%eax, %edi
	call	toupper@PLT
	movl	%eax, -124(%rbp)
	movl	-124(%rbp), %eax
	movl	%eax, %edx
	movl	-132(%rbp), %eax
	cltq
	movb	%dl, -112(%rbp,%rax)
	movq	$14, -120(%rbp)
	jmp	.L27
.L12:
	movl	$0, -136(%rbp)
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$32, -120(%rbp)
	jmp	.L27
.L14:
	movl	-132(%rbp), %eax
	cltq
	movb	$52, -112(%rbp,%rax)
	movq	$6, -120(%rbp)
	jmp	.L27
.L7:
	call	getchar@PLT
	movl	%eax, -128(%rbp)
	movl	-128(%rbp), %eax
	movb	%al, -137(%rbp)
	movq	$29, -120(%rbp)
	jmp	.L27
.L16:
	movq	$6, -120(%rbp)
	jmp	.L27
.L22:
	movl	-132(%rbp), %eax
	cltq
	movzbl	-112(%rbp,%rax), %eax
	movsbl	%al, %eax
	movl	%eax, %edi
	call	putchar@PLT
	addl	$1, -132(%rbp)
	movq	$0, -120(%rbp)
	jmp	.L27
.L10:
	movl	-132(%rbp), %eax
	cltq
	movb	$51, -112(%rbp,%rax)
	movq	$6, -120(%rbp)
	jmp	.L27
.L13:
	movq	$24, -120(%rbp)
	jmp	.L27
.L23:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$0, -132(%rbp)
	movq	$0, -120(%rbp)
	jmp	.L27
.L19:
	movl	-132(%rbp), %eax
	cltq
	movb	$53, -112(%rbp,%rax)
	movq	$6, -120(%rbp)
	jmp	.L27
.L26:
	movl	-132(%rbp), %eax
	cmpl	-136(%rbp), %eax
	jge	.L37
	movq	$3, -120(%rbp)
	jmp	.L27
.L37:
	movq	$12, -120(%rbp)
	jmp	.L27
.L21:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L43
	jmp	.L44
.L9:
	cmpb	$10, -137(%rbp)
	je	.L40
	movq	$25, -120(%rbp)
	jmp	.L27
.L40:
	movq	$5, -120(%rbp)
	jmp	.L27
.L45:
	nop
.L27:
	jmp	.L42
.L44:
	call	__stack_chk_fail@PLT
.L43:
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

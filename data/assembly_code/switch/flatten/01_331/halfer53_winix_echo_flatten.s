	.file	"halfer53_winix_echo_flatten.c"
	.text
	.globl	_TIG_IZ_e442_envp
	.bss
	.align 8
	.type	_TIG_IZ_e442_envp, @object
	.size	_TIG_IZ_e442_envp, 8
_TIG_IZ_e442_envp:
	.zero	8
	.globl	_TIG_IZ_e442_argv
	.align 8
	.type	_TIG_IZ_e442_argv, @object
	.size	_TIG_IZ_e442_argv, 8
_TIG_IZ_e442_argv:
	.zero	8
	.globl	_TIG_IZ_e442_argc
	.align 4
	.type	_TIG_IZ_e442_argc, @object
	.size	_TIG_IZ_e442_argc, 4
_TIG_IZ_e442_argc:
	.zero	4
	.local	buffer
	.comm	buffer,512,32
	.section	.rodata
.LC0:
	.string	"%s"
	.text
	.globl	main
	.type	main, @function
main:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$112, %rsp
	movl	%edi, -84(%rbp)
	movq	%rsi, -96(%rbp)
	movq	%rdx, -104(%rbp)
	movl	$0, -64(%rbp)
	jmp	.L2
.L3:
	movl	-64(%rbp), %eax
	cltq
	leaq	buffer(%rip), %rdx
	movb	$0, (%rax,%rdx)
	addl	$1, -64(%rbp)
.L2:
	cmpl	$511, -64(%rbp)
	jle	.L3
	nop
.L4:
	movq	$0, _TIG_IZ_e442_envp(%rip)
	nop
.L5:
	movq	$0, _TIG_IZ_e442_argv(%rip)
	nop
.L6:
	movl	$0, _TIG_IZ_e442_argc(%rip)
	nop
	nop
.L7:
.L8:
#APP
# 146 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-e442--0
# 0 "" 2
#NO_APP
	movl	-84(%rbp), %eax
	movl	%eax, _TIG_IZ_e442_argc(%rip)
	movq	-96(%rbp), %rax
	movq	%rax, _TIG_IZ_e442_argv(%rip)
	movq	-104(%rbp), %rax
	movq	%rax, _TIG_IZ_e442_envp(%rip)
	nop
	movq	$7, -40(%rbp)
.L45:
	cmpq	$26, -40(%rbp)
	ja	.L46
	movq	-40(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L11(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L11(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L11:
	.long	.L46-.L11
	.long	.L29-.L11
	.long	.L46-.L11
	.long	.L28-.L11
	.long	.L46-.L11
	.long	.L27-.L11
	.long	.L46-.L11
	.long	.L26-.L11
	.long	.L25-.L11
	.long	.L24-.L11
	.long	.L46-.L11
	.long	.L46-.L11
	.long	.L23-.L11
	.long	.L22-.L11
	.long	.L21-.L11
	.long	.L20-.L11
	.long	.L19-.L11
	.long	.L18-.L11
	.long	.L46-.L11
	.long	.L17-.L11
	.long	.L16-.L11
	.long	.L46-.L11
	.long	.L15-.L11
	.long	.L14-.L11
	.long	.L13-.L11
	.long	.L12-.L11
	.long	.L10-.L11
	.text
.L12:
	movl	-60(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-96(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movq	%rax, -56(%rbp)
	movq	$26, -40(%rbp)
	jmp	.L30
.L21:
	movq	-48(%rbp), %rax
	movq	%rax, -16(%rbp)
	addq	$1, -48(%rbp)
	movq	-16(%rbp), %rax
	movzbl	-65(%rbp), %edx
	movb	%dl, (%rax)
	addq	$1, -56(%rbp)
	movq	$5, -40(%rbp)
	jmp	.L30
.L20:
	movb	$9, -65(%rbp)
	movq	$14, -40(%rbp)
	jmp	.L30
.L23:
	leaq	buffer(%rip), %rax
	movq	%rax, -48(%rbp)
	movq	$1, -40(%rbp)
	jmp	.L30
.L25:
	movb	$10, -65(%rbp)
	movq	$14, -40(%rbp)
	jmp	.L30
.L29:
	movl	-60(%rbp), %eax
	cmpl	-84(%rbp), %eax
	jge	.L31
	movq	$25, -40(%rbp)
	jmp	.L30
.L31:
	movq	$24, -40(%rbp)
	jmp	.L30
.L14:
	movl	$0, %eax
	jmp	.L33
.L28:
	cmpl	$1, -84(%rbp)
	jg	.L34
	movq	$17, -40(%rbp)
	jmp	.L30
.L34:
	movq	$12, -40(%rbp)
	jmp	.L30
.L19:
	cmpb	$92, -65(%rbp)
	jne	.L36
	movq	$20, -40(%rbp)
	jmp	.L30
.L36:
	movq	$14, -40(%rbp)
	jmp	.L30
.L13:
	movq	-48(%rbp), %rax
	movq	%rax, -32(%rbp)
	addq	$1, -48(%rbp)
	movq	-32(%rbp), %rax
	movb	$10, (%rax)
	movq	-48(%rbp), %rax
	movq	%rax, -24(%rbp)
	addq	$1, -48(%rbp)
	movq	-24(%rbp), %rax
	movb	$0, (%rax)
	leaq	buffer(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$23, -40(%rbp)
	jmp	.L30
.L10:
	cmpl	$1, -60(%rbp)
	jle	.L38
	movq	$13, -40(%rbp)
	jmp	.L30
.L38:
	movq	$5, -40(%rbp)
	jmp	.L30
.L24:
	movq	-48(%rbp), %rax
	movb	$0, (%rax)
	addl	$1, -60(%rbp)
	movq	$1, -40(%rbp)
	jmp	.L30
.L22:
	movq	-48(%rbp), %rax
	movq	%rax, -8(%rbp)
	addq	$1, -48(%rbp)
	movq	-8(%rbp), %rax
	movb	$32, (%rax)
	movq	$5, -40(%rbp)
	jmp	.L30
.L17:
	movq	-56(%rbp), %rax
	movzbl	(%rax), %eax
	movb	%al, -65(%rbp)
	movq	$16, -40(%rbp)
	jmp	.L30
.L18:
	movl	$1, %eax
	jmp	.L33
.L15:
	movsbl	-65(%rbp), %eax
	cmpl	$110, %eax
	je	.L40
	cmpl	$116, %eax
	jne	.L41
	movq	$15, -40(%rbp)
	jmp	.L42
.L40:
	movq	$8, -40(%rbp)
	jmp	.L42
.L41:
	movq	$14, -40(%rbp)
	nop
.L42:
	jmp	.L30
.L27:
	movq	-56(%rbp), %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	je	.L43
	movq	$19, -40(%rbp)
	jmp	.L30
.L43:
	movq	$9, -40(%rbp)
	jmp	.L30
.L26:
	movl	$1, -60(%rbp)
	movq	$3, -40(%rbp)
	jmp	.L30
.L16:
	addq	$1, -56(%rbp)
	movq	-56(%rbp), %rax
	movzbl	(%rax), %eax
	movb	%al, -65(%rbp)
	movq	$22, -40(%rbp)
	jmp	.L30
.L46:
	nop
.L30:
	jmp	.L45
.L33:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
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

	.file	"Steph974-Git_holbertonschool-low_level_programming_4-print_alphabt_flatten.c"
	.text
	.globl	_TIG_IZ_Olxa_argc
	.bss
	.align 4
	.type	_TIG_IZ_Olxa_argc, @object
	.size	_TIG_IZ_Olxa_argc, 4
_TIG_IZ_Olxa_argc:
	.zero	4
	.globl	_TIG_IZ_Olxa_envp
	.align 8
	.type	_TIG_IZ_Olxa_envp, @object
	.size	_TIG_IZ_Olxa_envp, 8
_TIG_IZ_Olxa_envp:
	.zero	8
	.globl	_TIG_IZ_Olxa_argv
	.align 8
	.type	_TIG_IZ_Olxa_argv, @object
	.size	_TIG_IZ_Olxa_argv, 8
_TIG_IZ_Olxa_argv:
	.zero	8
	.text
	.globl	main
	.type	main, @function
main:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movl	%edi, -20(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movq	$0, _TIG_IZ_Olxa_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_Olxa_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_Olxa_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 88 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-Olxa--0
# 0 "" 2
#NO_APP
	movl	-20(%rbp), %eax
	movl	%eax, _TIG_IZ_Olxa_argc(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, _TIG_IZ_Olxa_argv(%rip)
	movq	-40(%rbp), %rax
	movq	%rax, _TIG_IZ_Olxa_envp(%rip)
	nop
	movq	$0, -8(%rbp)
.L24:
	cmpq	$9, -8(%rbp)
	ja	.L26
	movq	-8(%rbp), %rax
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
	.long	.L15-.L8
	.long	.L26-.L8
	.long	.L26-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L13:
	cmpb	$122, -9(%rbp)
	jg	.L16
	movq	$7, -8(%rbp)
	jmp	.L18
.L16:
	movq	$5, -8(%rbp)
	jmp	.L18
.L9:
	movsbl	-9(%rbp), %eax
	movl	%eax, %edi
	call	putchar@PLT
	movq	$9, -8(%rbp)
	jmp	.L18
.L14:
	movl	$0, %eax
	jmp	.L25
.L7:
	movzbl	-9(%rbp), %eax
	addl	$1, %eax
	movb	%al, -9(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L18
.L11:
	cmpb	$101, -9(%rbp)
	je	.L20
	movq	$8, -8(%rbp)
	jmp	.L18
.L20:
	movq	$9, -8(%rbp)
	jmp	.L18
.L12:
	movl	$10, %edi
	call	putchar@PLT
	movq	$3, -8(%rbp)
	jmp	.L18
.L15:
	movb	$97, -9(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L18
.L10:
	cmpb	$113, -9(%rbp)
	je	.L22
	movq	$6, -8(%rbp)
	jmp	.L18
.L22:
	movq	$9, -8(%rbp)
	jmp	.L18
.L26:
	nop
.L18:
	jmp	.L24
.L25:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
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

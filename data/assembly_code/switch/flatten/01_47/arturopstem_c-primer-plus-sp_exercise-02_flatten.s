	.file	"arturopstem_c-primer-plus-sp_exercise-02_flatten.c"
	.text
	.globl	_TIG_IZ_WFcI_argv
	.bss
	.align 8
	.type	_TIG_IZ_WFcI_argv, @object
	.size	_TIG_IZ_WFcI_argv, 8
_TIG_IZ_WFcI_argv:
	.zero	8
	.globl	_TIG_IZ_WFcI_envp
	.align 8
	.type	_TIG_IZ_WFcI_envp, @object
	.size	_TIG_IZ_WFcI_envp, 8
_TIG_IZ_WFcI_envp:
	.zero	8
	.globl	_TIG_IZ_WFcI_argc
	.align 4
	.type	_TIG_IZ_WFcI_argc, @object
	.size	_TIG_IZ_WFcI_argc, 4
_TIG_IZ_WFcI_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"Bye!"
.LC1:
	.string	"Enter text (# to quit):"
.LC2:
	.string	", "
.LC3:
	.string	"'\\t' =%4d"
.LC4:
	.string	"'\\n' =%4d"
.LC5:
	.string	" '%c' =%4d"
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
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	$0, _TIG_IZ_WFcI_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_WFcI_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_WFcI_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 102 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-WFcI--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_WFcI_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_WFcI_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_WFcI_envp(%rip)
	nop
	movq	$10, -8(%rbp)
.L38:
	cmpq	$24, -8(%rbp)
	ja	.L40
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
	.long	.L24-.L8
	.long	.L23-.L8
	.long	.L22-.L8
	.long	.L21-.L8
	.long	.L40-.L8
	.long	.L40-.L8
	.long	.L20-.L8
	.long	.L40-.L8
	.long	.L40-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L40-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L40-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L40-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L40-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L15:
	cmpb	$35, -17(%rbp)
	je	.L25
	movq	$19, -8(%rbp)
	jmp	.L27
.L25:
	movq	$12, -8(%rbp)
	jmp	.L27
.L17:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$6, -8(%rbp)
	jmp	.L27
.L23:
	movl	$0, -16(%rbp)
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$23, -8(%rbp)
	jmp	.L27
.L9:
	call	getchar@PLT
	movl	%eax, -12(%rbp)
	movl	-12(%rbp), %eax
	movb	%al, -17(%rbp)
	movq	$15, -8(%rbp)
	jmp	.L27
.L21:
	movl	-16(%rbp), %eax
	andl	$7, %eax
	testl	%eax, %eax
	jne	.L28
	movq	$24, -8(%rbp)
	jmp	.L27
.L28:
	movq	$16, -8(%rbp)
	jmp	.L27
.L14:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$23, -8(%rbp)
	jmp	.L27
.L7:
	movl	$10, %edi
	call	putchar@PLT
	movq	$23, -8(%rbp)
	jmp	.L27
.L19:
	movsbl	-17(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$22, -8(%rbp)
	jmp	.L27
.L16:
	movsbl	-17(%rbp), %eax
	cmpl	$9, %eax
	je	.L30
	cmpl	$10, %eax
	jne	.L31
	movq	$0, -8(%rbp)
	jmp	.L32
.L30:
	movq	$9, -8(%rbp)
	jmp	.L32
.L31:
	movq	$20, -8(%rbp)
	nop
.L32:
	jmp	.L27
.L12:
	cmpb	$0, -17(%rbp)
	jns	.L33
	movq	$23, -8(%rbp)
	jmp	.L27
.L33:
	movq	$2, -8(%rbp)
	jmp	.L27
.L13:
	movl	$0, -16(%rbp)
	movl	$10, %edi
	call	putchar@PLT
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$23, -8(%rbp)
	jmp	.L27
.L20:
	movl	$0, %eax
	jmp	.L39
.L10:
	cmpb	$10, -17(%rbp)
	jne	.L36
	movq	$17, -8(%rbp)
	jmp	.L27
.L36:
	movq	$3, -8(%rbp)
	jmp	.L27
.L18:
	movq	$1, -8(%rbp)
	jmp	.L27
.L24:
	movsbl	-17(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$22, -8(%rbp)
	jmp	.L27
.L22:
	addl	$1, -16(%rbp)
	movq	$13, -8(%rbp)
	jmp	.L27
.L11:
	movsbl	-17(%rbp), %edx
	movsbl	-17(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$22, -8(%rbp)
	jmp	.L27
.L40:
	nop
.L27:
	jmp	.L38
.L39:
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

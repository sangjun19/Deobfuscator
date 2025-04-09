	.file	"Sasmithere_C-practice-codes_7_flatten.c"
	.text
	.globl	_TIG_IZ_poyu_argv
	.bss
	.align 8
	.type	_TIG_IZ_poyu_argv, @object
	.size	_TIG_IZ_poyu_argv, 8
_TIG_IZ_poyu_argv:
	.zero	8
	.globl	_TIG_IZ_poyu_envp
	.align 8
	.type	_TIG_IZ_poyu_envp, @object
	.size	_TIG_IZ_poyu_envp, 8
_TIG_IZ_poyu_envp:
	.zero	8
	.globl	_TIG_IZ_poyu_argc
	.align 4
	.type	_TIG_IZ_poyu_argc, @object
	.size	_TIG_IZ_poyu_argc, 4
_TIG_IZ_poyu_argc:
	.zero	4
	.text
	.globl	is_substring
	.type	is_substring, @function
is_substring:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	$1, -8(%rbp)
.L6:
	cmpq	$0, -8(%rbp)
	je	.L2
	cmpq	$1, -8(%rbp)
	jne	.L8
	movq	-32(%rbp), %rdx
	movq	-24(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strstr@PLT
	movq	%rax, -16(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L4
.L2:
	cmpq	$0, -16(%rbp)
	setne	%al
	movzbl	%al, %eax
	jmp	.L7
.L8:
	nop
.L4:
	jmp	.L6
.L7:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	is_substring, .-is_substring
	.section	.rodata
.LC0:
	.string	"False"
.LC1:
	.string	"True"
	.text
	.globl	main
	.type	main, @function
main:
.LFB7:
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
	movq	$0, _TIG_IZ_poyu_envp(%rip)
	nop
.L10:
	movq	$0, _TIG_IZ_poyu_argv(%rip)
	nop
.L11:
	movl	$0, _TIG_IZ_poyu_argc(%rip)
	nop
	nop
.L12:
.L13:
#APP
# 89 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-poyu--0
# 0 "" 2
#NO_APP
	movl	-68(%rbp), %eax
	movl	%eax, _TIG_IZ_poyu_argc(%rip)
	movq	-80(%rbp), %rax
	movq	%rax, _TIG_IZ_poyu_argv(%rip)
	movq	-88(%rbp), %rax
	movq	%rax, _TIG_IZ_poyu_envp(%rip)
	nop
	movq	$3, -48(%rbp)
.L32:
	cmpq	$10, -48(%rbp)
	ja	.L35
	movq	-48(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L16(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L16(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L16:
	.long	.L35-.L16
	.long	.L25-.L16
	.long	.L24-.L16
	.long	.L23-.L16
	.long	.L22-.L16
	.long	.L21-.L16
	.long	.L20-.L16
	.long	.L19-.L16
	.long	.L18-.L16
	.long	.L17-.L16
	.long	.L15-.L16
	.text
.L22:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$1, -48(%rbp)
	jmp	.L26
.L18:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$5, -48(%rbp)
	jmp	.L26
.L25:
	movb	$112, -29(%rbp)
	movb	$121, -28(%rbp)
	movb	$116, -27(%rbp)
	movb	$104, -26(%rbp)
	movb	$111, -25(%rbp)
	movb	$110, -24(%rbp)
	movb	$0, -23(%rbp)
	leaq	-29(%rbp), %rdx
	leaq	-22(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	is_substring
	movl	%eax, -52(%rbp)
	movq	$6, -48(%rbp)
	jmp	.L26
.L23:
	movq	$7, -48(%rbp)
	jmp	.L26
.L17:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$1, -48(%rbp)
	jmp	.L26
.L20:
	cmpl	$0, -52(%rbp)
	je	.L27
	movq	$8, -48(%rbp)
	jmp	.L26
.L27:
	movq	$2, -48(%rbp)
	jmp	.L26
.L21:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L33
	jmp	.L34
.L15:
	cmpl	$0, -56(%rbp)
	je	.L30
	movq	$9, -48(%rbp)
	jmp	.L26
.L30:
	movq	$4, -48(%rbp)
	jmp	.L26
.L19:
	movb	$72, -22(%rbp)
	movb	$101, -21(%rbp)
	movb	$108, -20(%rbp)
	movb	$108, -19(%rbp)
	movb	$111, -18(%rbp)
	movb	$44, -17(%rbp)
	movb	$32, -16(%rbp)
	movb	$119, -15(%rbp)
	movb	$111, -14(%rbp)
	movb	$114, -13(%rbp)
	movb	$108, -12(%rbp)
	movb	$100, -11(%rbp)
	movb	$33, -10(%rbp)
	movb	$0, -9(%rbp)
	movb	$119, -35(%rbp)
	movb	$111, -34(%rbp)
	movb	$114, -33(%rbp)
	movb	$108, -32(%rbp)
	movb	$100, -31(%rbp)
	movb	$0, -30(%rbp)
	leaq	-35(%rbp), %rdx
	leaq	-22(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	is_substring
	movl	%eax, -56(%rbp)
	movq	$10, -48(%rbp)
	jmp	.L26
.L24:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$5, -48(%rbp)
	jmp	.L26
.L35:
	nop
.L26:
	jmp	.L32
.L34:
	call	__stack_chk_fail@PLT
.L33:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
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
